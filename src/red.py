import time

import numpy as np
import torch
from torch import nn

ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class MLP(nn.Module):
    def __init__(self, n_entradas, capas, activacion, dropout):
        super().__init__()
        bloques, n = [], n_entradas
        for h in capas:
            bloques += [nn.Linear(n, h), ACTIVACIONES[activacion]()]
            if dropout > 0:
                bloques.append(nn.Dropout(dropout))
            n = h
        bloques.append(nn.Linear(n, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x).squeeze(-1)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def crear_optimizador(modelo, hp):
    if hp["optimizador"] == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=hp["lr"], weight_decay=hp["l2"])
    return torch.optim.SGD(modelo.parameters(), lr=hp["lr"], momentum=0.9, weight_decay=hp["l2"])


def bce(z, y, reduction="mean"):
    return nn.functional.binary_cross_entropy_with_logits(z, y, reduction=reduction)


def mse(z, y, reduction="mean"):
    return nn.functional.mse_loss(torch.sigmoid(z), y, reduction=reduction)


PERDIDAS = {"bce": bce, "mse": mse}


def perdida_promedio(modelo, X, y, f=bce, tam=8192):
    modelo.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), tam):
            total += f(modelo(X[i:i + tam]), y[i:i + tam], reduction="sum").item()
    return total / len(X)


def entrenar(X_tr, y_tr, X_va, y_va, hp, semilla, max_epocas, paciencia):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    X_tr, y_tr = torch.tensor(np.asarray(X_tr), dtype=torch.float32), torch.tensor(np.asarray(y_tr), dtype=torch.float32)
    X_va, y_va = torch.tensor(np.asarray(X_va), dtype=torch.float32), torch.tensor(np.asarray(y_va), dtype=torch.float32)
    modelo = MLP(X_tr.shape[1], hp["capas"], hp["activacion"], hp["dropout"])
    opt = crear_optimizador(modelo, hp)
    f = PERDIDAS[hp.get("perdida", "bce")]
    historia, mejor, mejor_estado, sin_mejora = [], np.inf, None, 0
    t0 = time.time()
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        orden = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(X_tr), hp["batch"]):
            idx = orden[i:i + hp["batch"]]
            opt.zero_grad()
            perdida = f(modelo(X_tr[idx]), y_tr[idx])
            perdida.backward()
            opt.step()
        p_tr, p_va = perdida_promedio(modelo, X_tr, y_tr, f), perdida_promedio(modelo, X_va, y_va, f)
        historia.append((epoca, p_tr, p_va))
        if p_va < mejor - 1e-5:
            mejor, sin_mejora = p_va, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, historia, time.time() - t0


class ModeloRed:
    def __init__(self, modelo, hp):
        self.hp = hp
        self.n_entradas = modelo.red[0].in_features
        self.estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
        self._red = None

    def red(self):
        if self._red is None:
            self._red = MLP(self.n_entradas, self.hp["capas"], self.hp["activacion"], self.hp["dropout"])
            self._red.load_state_dict(self.estado)
            self._red.eval()
        return self._red

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_red"] = None
        return d

    def logit(self, X):
        with torch.no_grad():
            return self.red()(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

    def predecir(self, X):
        return 1 / (1 + np.exp(-self.logit(X)))

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.logit(X)))
        return np.column_stack([1 - p, p])

    def gradientes_integrados(self, X, x0, pasos=64):
        red = self.red()
        X = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)
        x0 = torch.as_tensor(np.asarray(x0), dtype=torch.float32)
        alphas = (torch.arange(pasos, dtype=torch.float32) + 0.5) / pasos
        total = torch.zeros_like(X)
        for a in alphas:
            puntos = (x0 + a * (X - x0)).requires_grad_(True)
            red(puntos).sum().backward()
            total += puntos.grad
        return ((X - x0) * total / pasos).numpy()
