from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]


def leer_yaml(ruta):
    with open(RAIZ / ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def aplicar_cambios(cfg, cambios):
    for cambio in cambios or []:
        clave, _, valor = cambio.partition("=")
        nodo = cfg
        partes = clave.split(".")
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = yaml.safe_load(valor)
    return cfg


def cargar_datos(cfg):
    cfg_datos = leer_yaml("config/datos.yaml")
    df = pd.read_csv(RAIZ / cfg["datos"], parse_dates=["issue_d"])
    df = df[df["issue_d"] >= cfg["fecha_inicio"]].copy()
    numericas = [c for c in cfg_datos["numericas"] + ["antiguedad_credito"] if c not in cfg["excluir"]]
    categoricas = [c for c in cfg_datos["categoricas"] if c not in cfg["excluir"]]
    return df, numericas, categoricas


def particionar(df, cfg):
    partes = {}
    for nombre, (ini, fin) in cfg["particion"].items():
        partes[nombre] = df[(df["issue_d"] >= ini) & (df["issue_d"] <= fin)].copy()
    return partes


def es_faltante(X):
    X = np.asarray(X, dtype=float)
    return np.isnan(X).astype(float)


def nombres_nunca(transformador, nombres):
    return np.asarray([f"nunca_{c.replace('mths_since_', '')}" for c in nombres], dtype=object)


class Recorte:
    def __init__(self, p_bajo, p_alto):
        self.p_bajo, self.p_alto = p_bajo, p_alto

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.bajo_ = np.nanpercentile(X, self.p_bajo, axis=0)
        self.alto_ = np.nanpercentile(X, self.p_alto, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.bajo_, self.alto_)

    def get_params(self, deep=True):
        return {"p_bajo": self.p_bajo, "p_alto": self.p_alto}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def get_feature_names_out(self, nombres=None):
        return np.asarray(nombres, dtype=object)


def construir_preprocesador(cfg, numericas, categoricas):
    con_log = [c for c in numericas if c in cfg["log1p"]]
    sin_log = [c for c in numericas if c not in cfg["log1p"]]
    nunca = [c for c in cfg["indicador_nunca"] if c in numericas]
    p1, p99 = cfg["recorte_percentiles"]

    def tubo(log):
        pasos = [("imputar", SimpleImputer(strategy="median"))]
        if log:
            pasos.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        pasos += [("recorte", Recorte(p1, p99)), ("escala", StandardScaler())]
        return Pipeline(pasos)

    return ColumnTransformer([
        ("num_log", tubo(True), con_log),
        ("num", tubo(False), sin_log),
        ("nunca", FunctionTransformer(es_faltante, feature_names_out=nombres_nunca), nunca),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=cfg["min_frecuencia_categoria"],
                              sparse_output=False), categoricas),
    ], verbose_feature_names_out=False)
