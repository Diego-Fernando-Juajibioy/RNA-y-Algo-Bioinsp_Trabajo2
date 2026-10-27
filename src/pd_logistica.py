import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from comun import RAIZ, aplicar_cambios, cargar_datos, construir_preprocesador, leer_yaml, particionar
from metricas import curva_calibracion, evaluar
from scorecard import puntaje_desde_prob, puntos_logistica

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#6b6b66"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pd.yaml")
    p.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = p.parse_args()
    cfg = aplicar_cambios(leer_yaml(args.config), args.set)
    sal = {k: RAIZ / v for k, v in cfg["salidas"].items()}
    for d in sal.values():
        d.mkdir(parents=True, exist_ok=True)

    df, numericas, categoricas = cargar_datos(cfg)
    partes = particionar(df, cfg)
    entradas = numericas + categoricas
    resumen_part = {k: {"n": len(v), "tasa_malos": round(float(v.malo.mean()), 4),
                        "desde": str(v.issue_d.min().date()), "hasta": str(v.issue_d.max().date())}
                    for k, v in partes.items()}
    for k, v in resumen_part.items():
        print(f"{k:14s} {v['n']:7d} créditos  {v['tasa_malos']:.1%} malos  ({v['desde']} a {v['hasta']})")

    prep = construir_preprocesador(cfg, numericas, categoricas)
    X_tr = prep.fit_transform(partes["entrenamiento"][entradas])
    nombres = list(prep.get_feature_names_out())
    y_tr = partes["entrenamiento"]["malo"].values

    t0 = time.time()
    modelo = LogisticRegression(C=cfg["logistica"]["C"], max_iter=cfg["logistica"]["max_iter"])
    modelo.fit(X_tr, y_tr)
    segundos = time.time() - t0
    n_param = modelo.coef_.size + modelo.intercept_.size
    print(f"\nRegresión logística: {X_tr.shape[1]} entradas, {n_param} parámetros, {segundos:.1f} s")

    resultados, probs = {}, {}
    for nombre, parte in partes.items():
        X = prep.transform(parte[entradas])
        probs[nombre] = modelo.predict_proba(X)[:, 1]
        resultados[nombre] = evaluar(parte["malo"].values, probs[nombre])
    tabla = pd.DataFrame(resultados).T
    print("\n" + tabla.round(4).to_string())

    sc = cfg["scorecard"]
    puntajes_tr = puntaje_desde_prob(probs["entrenamiento"], sc["puntaje_base"], sc["odds_base"], sc["pdo"])
    base, puntos = puntos_logistica(modelo, nombres, X_tr, sc["puntaje_base"], sc["odds_base"], sc["pdo"])
    rango = pd.DataFrame({"variable": nombres, "coeficiente": modelo.coef_[0],
                          "puntos_min": [puntos[n].min() for n in nombres],
                          "puntos_max": [puntos[n].max() for n in nombres]})
    rango["amplitud"] = rango.puntos_max - rango.puntos_min
    rango.sort_values("amplitud", ascending=False).to_csv(sal["tablas"] / "scorecard_logistica_puntos.csv",
                                                           index=False, float_format="%.3f")

    tr = partes["entrenamiento"]
    perfil = {c: float(tr[c].median()) for c in numericas}
    perfil.update({c: tr[c].mode().iloc[0] for c in categoricas})
    rangos = {c: (float(tr[c].quantile(0.01)), float(tr[c].quantile(0.99))) for c in numericas}
    opciones = {c: tr[c].value_counts().index.tolist() for c in categoricas}
    tasa_por_grado = tr.groupby("grade")["int_rate"].median().round(2).to_dict()

    joblib.dump({"preprocesador": prep, "modelo": modelo, "entradas": entradas, "nombres": nombres,
                 "numericas": numericas, "categoricas": categoricas, "scorecard": sc,
                 "puntajes_entrenamiento": np.sort(puntajes_tr).astype(np.float32),
                 "perfil": perfil, "rangos": rangos, "opciones": opciones, "tasa_por_grado": tasa_por_grado,
                 "nombre_modelo": "Regresión logística"},
                sal["modelos"] / "pd_logistica.joblib")
    json.dump({"particiones": resumen_part, "metricas": tabla.round(5).to_dict(orient="index"),
               "parametros_entrenables": int(n_param), "segundos_entrenamiento": round(segundos, 2),
               "configuraciones_probadas": 1, "puntos_base": round(float(base), 2), "scorecard": sc},
              open(sal["tablas"] / "metricas_pd_logistica.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for (nombre, color, ls) in [("validacion", AZUL, "-"), ("prueba", NARANJA, "--")]:
        pred, obs = curva_calibracion(partes[nombre]["malo"].values, probs[nombre])
        axes[0].plot(pred * 100, obs * 100, color=color, ls=ls, marker="o", ms=4, lw=2, label=nombre)
    lim = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, lim], [0, lim], color=GRIS, lw=0.8, ls=(0, (2, 2)), label="calibración perfecta")
    axes[0].set_xlabel("Probabilidad predicha (%)"); axes[0].set_ylabel("Tasa observada de malos (%)")
    axes[0].set_title("Curva de calibración (deciles)"); axes[0].legend(frameon=False, fontsize=7)
    s_te = puntaje_desde_prob(probs["prueba"], sc["puntaje_base"], sc["odds_base"], sc["pdo"])
    y_te = partes["prueba"]["malo"].values
    bins = np.linspace(np.percentile(s_te, 0.5), np.percentile(s_te, 99.5), 40)
    axes[1].hist(s_te[y_te == 0], bins=bins, density=True, alpha=0.55, color=AZUL, label="buenos")
    axes[1].hist(s_te[y_te == 1], bins=bins, density=True, alpha=0.55, color=NARANJA, label="malos")
    axes[1].set_xlabel("Puntaje"); axes[1].set_ylabel("Densidad")
    axes[1].set_title("Puntaje en prueba (2015) por desenlace"); axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig05_logistica_calibracion_puntaje.png", dpi=160); plt.close(fig)
    print(f"\nModelo en {sal['modelos'] / 'pd_logistica.joblib'}")


if __name__ == "__main__":
    main()
