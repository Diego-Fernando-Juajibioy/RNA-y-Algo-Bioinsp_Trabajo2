import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks(y, p):
    orden = np.argsort(p)
    y = np.asarray(y)[orden]
    acum_malos = np.cumsum(y) / y.sum()
    acum_buenos = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(acum_malos - acum_buenos)))


def curva_calibracion(y, p, n_bins=10):
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    grupo = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, n_bins - 1)
    pred = np.array([p[grupo == k].mean() for k in range(n_bins)])
    obs = np.array([np.asarray(y)[grupo == k].mean() for k in range(n_bins)])
    return pred, obs


def evaluar(y, p):
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks(y, p), "Brier": brier_score_loss(y, p),
            "tasa_observada": float(np.mean(y)), "prob_media": float(np.mean(p))}
