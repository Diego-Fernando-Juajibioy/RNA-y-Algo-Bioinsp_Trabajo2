import numpy as np


def parametros(puntaje_base, odds_base, pdo):
    factor = pdo / np.log(2)
    offset = puntaje_base - factor * np.log(odds_base)
    return factor, offset


def puntaje_desde_prob(p, puntaje_base, odds_base, pdo):
    factor, offset = parametros(puntaje_base, odds_base, pdo)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return offset + factor * np.log((1 - p) / p)


def prob_desde_puntaje(s, puntaje_base, odds_base, pdo):
    factor, offset = parametros(puntaje_base, odds_base, pdo)
    return 1 / (1 + np.exp((s - offset) / factor))


def puntos_logistica(modelo, nombres, X, puntaje_base, odds_base, pdo):
    factor, offset = parametros(puntaje_base, odds_base, pdo)
    puntos = -factor * X * modelo.coef_[0]
    base = offset - factor * modelo.intercept_[0]
    return base, dict(zip(nombres, puntos.T))
