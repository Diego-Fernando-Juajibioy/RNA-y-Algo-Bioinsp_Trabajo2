import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import RAIZ, aplicar_cambios, cargar_datos, leer_yaml, particionar

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"


def predecir(paquete, df):
    X = paquete["preprocesador"].transform(df[paquete["entradas"]])
    m = paquete["modelo"]
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predecir(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/lgd.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = ap.parse_args()
    cfg = aplicar_cambios(leer_yaml(args.config), args.set)
    cfg_pd = leer_yaml("config/pd.yaml")
    sal = {k: RAIZ / v for k, v in cfg["salidas"].items()}
    mod = sal["modelos"]

    df, _, _ = cargar_datos(cfg_pd)
    prueba = particionar(df, cfg_pd)["prueba"]
    post = pd.read_csv(RAIZ / cfg["datos_post"], usecols=["id", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"])
    prueba = prueba.merge(post, on="id", how="left")

    pd_red, pd_log = joblib.load(mod / "pd_red.joblib"), joblib.load(mod / "pd_logistica.joblib")
    lgd_red, lgd_lin = joblib.load(mod / "lgd_red.joblib"), joblib.load(mod / "lgd_lineal.joblib")
    factor = lgd_red["ead_sobre_monto"] if cfg["perdida_esperada"]["ead"] == "ajustada" else 1.0
    ead = prueba["loan_amnt"].values * factor

    perdida_obs = np.where(prueba["malo"] == 1,
                           (prueba["funded_amnt"] - prueba["total_rec_prncp"] - (prueba["recoveries"] - prueba["collection_recovery_fee"])).clip(lower=0),
                           0.0)
    combinaciones = {"Red (PD) × Red (LGD)": (pd_red, lgd_red), "Logística (PD) × Lineal (LGD)": (pd_log, lgd_lin)}
    filas, el = [], {}
    for nombre, (m_pd, m_lgd) in combinaciones.items():
        p, l = predecir(m_pd, prueba), predecir(m_lgd, prueba)
        el[nombre] = p * l * ead
        filas.append({"combinacion": nombre, "PD_media": p.mean(), "LGD_media": l.mean(), "EAD_total": ead.sum(),
                      "perdida_esperada_total": el[nombre].sum(), "tasa_perdida_esperada": el[nombre].sum() / prueba["loan_amnt"].sum()})
    filas.append({"combinacion": "Observado (aproximado)", "PD_media": prueba["malo"].mean(), "LGD_media": np.nan,
                  "EAD_total": np.nan, "perdida_esperada_total": perdida_obs.sum(), "tasa_perdida_esperada": perdida_obs.sum() / prueba["loan_amnt"].sum()})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(sal["tablas"] / "perdida_esperada_prueba.csv", index=False, float_format="%.4f")
    print(f"Créditos de prueba (2015): {len(prueba)}  ·  EAD = monto pedido × {factor:.3f}")
    print(tabla.round(4).to_string(index=False))

    principal = list(combinaciones)[0]
    tasa = el[principal] / prueba["loan_amnt"].values
    dec = pd.qcut(tasa, 10, labels=False)
    por_dec = pd.DataFrame({"decil": dec, "esperada": el[principal], "observada": perdida_obs, "monto": prueba["loan_amnt"].values})
    por_dec = por_dec.groupby("decil").sum()
    por_grado = pd.DataFrame({"grado": prueba["grade"].values, **{k: v for k, v in el.items()}, "observada": perdida_obs,
                              "monto": prueba["loan_amnt"].values}).groupby("grado").sum()
    por_grado.div(por_grado["monto"], axis=0).drop(columns="monto").to_csv(sal["tablas"] / "perdida_esperada_por_grado.csv", float_format="%.4f")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(por_dec.index + 1, por_dec["esperada"] / por_dec["monto"] * 100, color=NARANJA, marker="o", lw=2, label="Esperada (red × red)")
    a1.plot(por_dec.index + 1, por_dec["observada"] / por_dec["monto"] * 100, color="#1a1a19", marker="s", lw=2, ls="--", label="Observada (aprox.)")
    a1.set_xticks(range(1, 11)); a1.set_xlabel("Decil de pérdida esperada (1 = menor)"); a1.set_ylabel("% del monto prestado")
    a1.set_title("Pérdida esperada vs. observada en prueba (2015)"); a1.legend(frameon=False, fontsize=7)
    x = np.arange(len(por_grado))
    ancho = 0.27
    for k, (col, color) in enumerate([(list(el)[0], NARANJA), (list(el)[1], AZUL), ("observada", "#3d3d3a")]):
        a2.bar(x + (k - 1) * ancho, por_grado[col] / por_grado["monto"] * 100, width=ancho, color=color,
               label=col if col != "observada" else "Observada (aprox.)")
    a2.set_xticks(x, por_grado.index); a2.set_xlabel("Grado"); a2.set_ylabel("% del monto prestado")
    a2.set_title("Por grado"); a2.legend(frameon=False, fontsize=7)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig13_perdida_esperada.png", dpi=160); plt.close(fig)
    print(f"Figura 13 en {sal['figuras']}")


if __name__ == "__main__":
    main()
