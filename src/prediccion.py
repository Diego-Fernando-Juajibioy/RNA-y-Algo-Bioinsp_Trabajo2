from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scorecard import prob_desde_puntaje, puntaje_desde_prob

RAIZ = Path(__file__).resolve().parents[1]


def cargar_modelo(nombre="pd_logistica.joblib"):
    return joblib.load(RAIZ / "models" / nombre)


def cuota_mensual(monto, tasa_anual, meses):
    r = tasa_anual / 1200
    return monto * r / (1 - (1 + r) ** (-meses))


def completar(paquete, datos):
    fila = dict(paquete["perfil"])
    fila.update(datos)
    meses = 60 if str(fila["term"]).startswith("60") else 36
    fila["installment"] = cuota_mensual(fila["loan_amnt"], fila["int_rate"], meses)
    return fila


def probabilidad(paquete, filas):
    df = pd.DataFrame(filas)[paquete["entradas"]]
    X = paquete["preprocesador"].transform(df)
    return paquete["modelo"].predict_proba(X)[:, 1]


def puntaje(paquete, p):
    sc = paquete["scorecard"]
    return puntaje_desde_prob(np.asarray(p), sc["puntaje_base"], sc["odds_base"], sc["pdo"])


def percentil(paquete, s):
    pob = paquete["puntajes_entrenamiento"]
    return float(np.searchsorted(pob, s, side="right") / len(pob) * 100)


def contribuciones(paquete, fila):
    modelo = paquete["modelo"]
    if not hasattr(modelo, "coef_"):
        return None
    sc = paquete["scorecard"]
    factor = sc["pdo"] / np.log(2)
    prep = paquete["preprocesador"]
    x = prep.transform(pd.DataFrame([fila])[paquete["entradas"]])[0]
    x0 = prep.transform(pd.DataFrame([paquete["perfil"]])[paquete["entradas"]])[0]
    aporte = -factor * modelo.coef_[0] * (x - x0)
    nombres = paquete["nombres"]
    por_variable = {}
    for v in paquete["entradas"]:
        idx = [i for i, n in enumerate(nombres) if n == v or n.startswith(v + "_") or n == "nunca_" + v.replace("mths_since_", "")]
        por_variable[v] = float(aporte[idx].sum())
    return pd.Series(por_variable).sort_values()


def curva_que_pasa_si(paquete, fila, variable, valores):
    filas = []
    for v in valores:
        f = dict(fila)
        f[variable] = v
        if variable in ("loan_amnt", "int_rate", "term"):
            meses = 60 if str(f["term"]).startswith("60") else 36
            f["installment"] = cuota_mensual(f["loan_amnt"], f["int_rate"], meses)
        filas.append(f)
    return puntaje(paquete, probabilidad(paquete, filas))


def prob_de_puntaje(paquete, s):
    sc = paquete["scorecard"]
    return prob_desde_puntaje(s, sc["puntaje_base"], sc["odds_base"], sc["pdo"])
