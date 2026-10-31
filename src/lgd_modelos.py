import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from comun import RAIZ, LinealRecortado, aplicar_cambios, cargar_datos, construir_preprocesador, leer_yaml, particionar
from metricas import evaluar_regresion
from pd_red import sortear_configuracion
from red import ModeloRed, contar_parametros, entrenar

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
METRICAS = ["MAE", "RMSE", "R2", "W1"]


def preparar(cfg, demo):
    df, numericas, categoricas = cargar_datos(cfg)
    partes = particionar(df, cfg)
    if demo:
        partes = {k: v.sample(frac=demo, random_state=0) for k, v in partes.items()}
    entradas = numericas + categoricas
    prep = construir_preprocesador(cfg, numericas, categoricas)
    prep.fit(partes["entrenamiento"][entradas])
    X = {k: prep.transform(v[entradas]).astype(np.float32) for k, v in partes.items()}
    y = {k: v["lgd"].values.astype(np.float32) for k, v in partes.items()}
    return partes, prep, X, y, numericas, categoricas


def lineal(cfg, X, y):
    t0, filas, mejor = time.time(), [], None
    for alfa in cfg["lineal"]["alfa_grid"]:
        m = LinealRecortado(Ridge(alpha=alfa).fit(X["entrenamiento"], y["entrenamiento"]))
        rmse = evaluar_regresion(y["validacion"], m.predecir(X["validacion"]))["RMSE"]
        filas.append({"alfa": alfa, "RMSE_validacion": round(rmse, 5)})
        print(f"  alfa = {alfa:<8} RMSE validación = {rmse:.4f}")
        if mejor is None or rmse < mejor[0]:
            mejor = (rmse, alfa, m)
    return mejor[2], mejor[1], filas, time.time() - t0


def busqueda(cfg, X, y, demo):
    r = cfg["red"]
    n = r["demo"]["n_configuraciones"] if demo else r["busqueda"]["n_configuraciones"]
    max_ep = r["demo"]["max_epocas"] if demo else r["max_epocas"]
    rng = np.random.default_rng(cfg["semilla"])
    filas, t0 = [], time.time()
    for i in range(n):
        hp = {**sortear_configuracion(rng, r["busqueda"]), "perdida": r["perdida"]}
        modelo, hist, seg = entrenar(X["entrenamiento"], y["entrenamiento"], X["validacion"], y["validacion"],
                                     hp, cfg["semilla"], max_ep, r["paciencia"])
        m = evaluar_regresion(y["validacion"], ModeloRed(modelo, hp).predecir(X["validacion"]))
        filas.append({"config": i + 1, **{k: str(v) if k == "capas" else v for k, v in hp.items()},
                      "parametros": contar_parametros(modelo), "epocas": len(hist), "segundos": round(seg, 1),
                      "RMSE_validacion": round(m["RMSE"], 5), "MAE_validacion": round(m["MAE"], 5)})
        print(f"  {i + 1:2d}/{n} capas={hp['capas']} {hp['activacion']:10s} drop={hp['dropout']:.1f} "
              f"l2={hp['l2']:.1e} {hp['optimizador']:4s} lr={hp['lr']:.1e} batch={hp['batch']:3d} -> "
              f"RMSE val {m['RMSE']:.4f} ({len(hist)} épocas, {seg:.0f} s)")
    tabla = pd.DataFrame(filas).sort_values("RMSE_validacion")
    mejor = tabla.iloc[0]
    hp = {"capas": json.loads(mejor["capas"]), "activacion": mejor["activacion"], "dropout": float(mejor["dropout"]),
          "l2": float(mejor["l2"]), "optimizador": mejor["optimizador"], "lr": float(mejor["lr"]),
          "batch": int(mejor["batch"]), "perdida": r["perdida"]}
    print(f"\nMejor configuración (config {mejor['config']}): {hp}  RMSE val {mejor['RMSE_validacion']:.4f}")
    return hp, tabla, time.time() - t0


def semillas(cfg, hp, X, y, demo):
    r = cfg["red"]
    n = r["demo"]["n_semillas"] if demo else r["final"]["n_semillas"]
    max_ep = r["demo"]["max_epocas"] if demo else r["max_epocas"]
    filas, mejor, t0 = [], None, time.time()
    for k in range(n):
        semilla = cfg["semilla"] + 1000 + k
        modelo, hist, seg = entrenar(X["entrenamiento"], y["entrenamiento"], X["validacion"], y["validacion"],
                                     hp, semilla, max_ep, r["paciencia"])
        envoltura = ModeloRed(modelo, hp)
        fila = {"semilla": semilla, "epocas": len(hist), "segundos": round(seg, 1)}
        for parte in ("validacion", "prueba"):
            m = evaluar_regresion(y[parte], envoltura.predecir(X[parte]))
            fila.update({f"{c}_{parte}": m[c] for c in METRICAS})
        filas.append(fila)
        print(f"  semilla {semilla}: RMSE val {fila['RMSE_validacion']:.4f}  RMSE prueba {fila['RMSE_prueba']:.4f}  "
              f"MAE prueba {fila['MAE_prueba']:.4f}  ({len(hist)} épocas, {seg:.0f} s)")
        if mejor is None or fila["RMSE_validacion"] < mejor[0]:
            mejor = (fila["RMSE_validacion"], semilla, envoltura, hist, contar_parametros(modelo))
    return pd.DataFrame(filas), mejor, time.time() - t0


def resumen(tabla):
    res = {}
    for c in tabla.columns:
        if c.split("_")[0] not in METRICAS:
            continue
        mayor_mejor = c.startswith("R2")
        res[c] = {"media": tabla[c].mean(), "desv": tabla[c].std(ddof=1),
                  "mejor": tabla[c].max() if mayor_mejor else tabla[c].min(),
                  "peor": tabla[c].min() if mayor_mejor else tabla[c].max()}
    return res


def figuras(sal, tabla_busq, tabla_sem, y, p_lin, p_red, p_media, hist, semilla):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for opt, color in [("adam", AZUL), ("sgd", NARANJA)]:
        t = tabla_busq[tabla_busq.optimizador == opt]
        ax.scatter(t.parametros, t.RMSE_validacion, color=color, s=28, label=opt.upper(), edgecolor="white", lw=0.5)
    m = tabla_busq.iloc[0]
    ax.scatter([m.parametros], [m.RMSE_validacion], s=140, facecolor="none", edgecolor="#1a1a19", lw=1.5, label="elegida")
    ax.axhline(p_lin["rmse_validacion"], color=VERDE, ls="--", lw=1.2, label="Regresión lineal")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Parámetros entrenables (escala log)"); ax.set_ylabel("RMSE en validación")
    ax.set_title(f"LGD: búsqueda aleatoria de {len(tabla_busq)} configuraciones"); ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig11_lgd_busqueda.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(y, bins=bins, color="#cfcfca", label="Observada", density=True)
    axes[0].hist(p_lin["prueba"], bins=bins, histtype="step", color=VERDE, lw=2, ls="--", label="Regresión lineal", density=True)
    axes[0].hist(p_red, bins=bins, histtype="step", color=NARANJA, lw=2, label="Red neuronal", density=True)
    axes[0].axvline(p_media, color=GRIS, lw=1, ls=(0, (2, 2)), label="Media de entrenamiento")
    axes[0].set_xlim(0.5, 1.02); axes[0].set_xlabel("LGD"); axes[0].set_ylabel("Densidad")
    axes[0].set_title("Distribución en prueba (2014)"); axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].boxplot([tabla_sem["RMSE_prueba"]], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor="#f6c7b2", edgecolor=NARANJA), medianprops=dict(color="#1a1a19"))
    axes[1].scatter(np.random.default_rng(0).normal(1, 0.04, len(tabla_sem)), tabla_sem["RMSE_prueba"], s=12, color=NARANJA, zorder=3)
    axes[1].axhline(p_lin["rmse_prueba"], color=VERDE, ls="--", lw=1.5, label="Regresión lineal")
    axes[1].set_xticks([1], [f"Red neuronal\n({len(tabla_sem)} semillas)"]); axes[1].set_ylabel("RMSE en prueba")
    axes[1].set_title("Variabilidad entre semillas"); axes[1].legend(frameon=False, fontsize=7)
    h = np.array(hist)
    axes[2].plot(h[:, 0], h[:, 1], color=AZUL, lw=2, label="entrenamiento")
    axes[2].plot(h[:, 0], h[:, 2], color=NARANJA, lw=2, ls="--", label="validación")
    axes[2].axvline(h[np.argmin(h[:, 2]), 0], color=GRIS, lw=0.8, ls=(0, (2, 2)), label="parada temprana")
    axes[2].set_xlabel("Época"); axes[2].set_ylabel("Pérdida")
    axes[2].set_title(f"Curva de aprendizaje (semilla {semilla})"); axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig12_lgd_resultados.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/lgd.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    ap.add_argument("--demo", action="store_true", help="muestra de los datos, menos de 2 minutos")
    args = ap.parse_args()
    cfg = aplicar_cambios(leer_yaml(args.config), args.set)
    sal = {k: RAIZ / v for k, v in cfg["salidas"].items()}
    for d in sal.values():
        d.mkdir(parents=True, exist_ok=True)
    demo = cfg["red"]["demo"]["fraccion"] if args.demo else None

    partes, prep, X, y, numericas, categoricas = preparar(cfg, demo)
    for k, v in partes.items():
        print(f"{k:14s} {len(v):6d} créditos castigados  LGD media {v.lgd.mean():.3f}  "
              f"({v.issue_d.min().date()} a {v.issue_d.max().date()})")
    print(f"{X['entrenamiento'].shape[1]} entradas" + ("  [MODO DEMO]" if demo else ""))

    media = float(y["entrenamiento"].mean())
    print("\nRegresión lineal (Ridge), alfa elegido por RMSE en validación:")
    lin, alfa, busq_lin, seg_lin = lineal(cfg, X, y)
    print(f"\nRed neuronal: búsqueda aleatoria con pérdida '{cfg['red']['perdida']}'")
    hp, tabla_busq, seg_busq = busqueda(cfg, X, y, demo)
    if cfg["red"]["final"]["hp"]:
        hp = cfg["red"]["final"]["hp"]
    print(f"\nEntrenando la configuración final con varias semillas: {hp}")
    tabla_sem, mejor, seg_sem = semillas(cfg, hp, X, y, demo)
    res = resumen(tabla_sem)

    filas = []
    for parte in ("validacion", "prueba"):
        m_media = evaluar_regresion(y[parte], np.full_like(y[parte], media))
        m_lin = evaluar_regresion(y[parte], lin.predecir(X[parte]))
        for c in METRICAS:
            filas.append({"conjunto": parte, "metrica": c, "modelo": "Media de entrenamiento", "media": m_media[c], "desv": 0.0, "mejor": m_media[c], "peor": m_media[c]})
            filas.append({"conjunto": parte, "metrica": c, "modelo": "Regresión lineal", "media": m_lin[c], "desv": 0.0, "mejor": m_lin[c], "peor": m_lin[c]})
            filas.append({"conjunto": parte, "metrica": c, "modelo": "Red neuronal", **res[f"{c}_{parte}"]})
    comp = pd.DataFrame(filas)
    print("\n" + comp.pivot_table(index=["conjunto", "metrica"], columns="modelo", values="media", sort=False).round(4).to_string())
    if demo:
        return

    _, semilla, envoltura, hist, n_param = mejor
    n_lin = lin.modelo.coef_.size + 1
    costo = pd.DataFrame([
        {"modelo": "Regresión lineal", "parametros": n_lin, "configuraciones": len(busq_lin), "segundos_totales": round(seg_lin, 2)},
        {"modelo": "Red neuronal", "parametros": n_param, "configuraciones": len(tabla_busq), "segundos_totales": round(seg_busq + seg_sem, 1)}])
    print("\n" + costo.to_string(index=False))

    tabla_busq.to_csv(sal["tablas"] / "busqueda_lgd_red.csv", index=False)
    tabla_sem.to_csv(sal["tablas"] / "semillas_lgd_red.csv", index=False, float_format="%.5f")
    comp.to_csv(sal["tablas"] / "comparacion_lgd.csv", index=False, float_format="%.4f")
    costo.to_csv(sal["tablas"] / "costo_lgd.csv", index=False)
    p_red = {k: envoltura.predecir(X[k]) for k in X}
    p_lin = {k: lin.predecir(X[k]) for k in X}
    cuantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    dist = pd.DataFrame({"Observada": np.quantile(y["prueba"], cuantiles), "Regresión lineal": np.quantile(p_lin["prueba"], cuantiles),
                         "Red neuronal": np.quantile(p_red["prueba"], cuantiles)}, index=[f"p{int(q * 100)}" for q in cuantiles])
    dist.loc["media"] = [y["prueba"].mean(), p_lin["prueba"].mean(), p_red["prueba"].mean()]
    dist.loc["desviación"] = [y["prueba"].std(), p_lin["prueba"].std(), p_red["prueba"].std()]
    dist.to_csv(sal["tablas"] / "distribucion_lgd_prueba.csv", float_format="%.4f")
    print("\n" + dist.round(4).to_string())

    json.dump({"alfa_elegido": alfa, "busqueda_alfa": busq_lin, "parametros_lineal": int(n_lin), "segundos_lineal": round(seg_lin, 2),
               "hp_red": hp, "parametros_red": int(n_param), "configuraciones_red": len(tabla_busq),
               "segundos_busqueda": round(seg_busq, 1), "segundos_semillas": round(seg_sem, 1), "semilla_desplegada": int(semilla),
               "media_entrenamiento": media, "resumen_red": {k: {kk: round(float(vv), 5) for kk, vv in v.items()} for k, v in res.items()},
               "lineal": {k: evaluar_regresion(y[k], p_lin[k]) for k in X}, "red_desplegada": {k: evaluar_regresion(y[k], p_red[k]) for k in X}},
              open(sal["tablas"] / "metricas_lgd.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    base = {"preprocesador": prep, "entradas": numericas + categoricas, "numericas": numericas, "categoricas": categoricas,
            "ead_sobre_monto": float((partes["entrenamiento"].ead_obs / partes["entrenamiento"].funded_amnt).mean())}
    joblib.dump({**base, "modelo": lin, "nombre_modelo": f"Regresión lineal (alfa = {alfa})"}, sal["modelos"] / "lgd_lineal.joblib")
    joblib.dump({**base, "modelo": envoltura, "nombre_modelo": f"Red neuronal {hp['capas']}", "hp": hp, "semilla": semilla},
                sal["modelos"] / "lgd_red.joblib")
    figuras(sal, tabla_busq, tabla_sem, y["prueba"],
            {"prueba": p_lin["prueba"], "rmse_validacion": evaluar_regresion(y["validacion"], p_lin["validacion"])["RMSE"],
             "rmse_prueba": evaluar_regresion(y["prueba"], p_lin["prueba"])["RMSE"]},
            p_red["prueba"], media, hist, semilla)
    print(f"\nModelos en {sal['modelos']} (red: semilla {semilla}, la mejor en validación). Figuras 11 y 12.")


if __name__ == "__main__":
    main()
