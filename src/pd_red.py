import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from comun import RAIZ, aplicar_cambios, cargar_datos, construir_preprocesador, leer_yaml, particionar
from metricas import curva_calibracion, evaluar
from red import ModeloRed, contar_parametros, entrenar
from scorecard import puntaje_desde_prob

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
METRICAS = ["AUC", "Gini", "KS", "Brier"]


def sortear_configuracion(rng, b):
    n_capas = int(rng.choice(b["capas_ocultas"]))
    primera = int(rng.choice(b["neuronas"]))
    capas = [max(8, primera // (2 ** k)) for k in range(n_capas)]
    opt = str(rng.choice(b["optimizador"]))
    lo, hi = b["lr_adam"] if opt == "adam" else b["lr_sgd"]
    return {"capas": capas, "activacion": str(rng.choice(b["activacion"])), "dropout": float(rng.choice(b["dropout"])),
            "l2": float(np.exp(rng.uniform(np.log(b["l2"][0]), np.log(b["l2"][1])))), "optimizador": opt,
            "lr": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), "batch": int(rng.choice(b["batch"]))}


def preparar(cfg_pd, demo):
    df, numericas, categoricas = cargar_datos(cfg_pd)
    partes = particionar(df, cfg_pd)
    if demo:
        partes = {k: v.sample(frac=demo, random_state=0) for k, v in partes.items()}
    entradas = numericas + categoricas
    prep = construir_preprocesador(cfg_pd, numericas, categoricas)
    prep.fit(partes["entrenamiento"][entradas])
    X = {k: prep.transform(v[entradas]).astype(np.float32) for k, v in partes.items()}
    y = {k: v["malo"].values.astype(np.float32) for k, v in partes.items()}
    return partes, prep, X, y, numericas, categoricas


def grafica_busqueda(tabla, ruta):
    mejor = tabla.loc[tabla.AUC_validacion.idxmax()]
    piso = max(tabla.AUC_validacion.quantile(0.1) - 0.004, tabla.AUC_validacion.min())
    fuera = tabla[tabla.AUC_validacion < piso]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for opt, color in [("adam", AZUL), ("sgd", NARANJA)]:
        t = tabla[(tabla.optimizador == opt) & (tabla.AUC_validacion >= piso)]
        ax.scatter(t.parametros, t.AUC_validacion, color=color, s=28, label=opt.upper(), edgecolor="white", lw=0.5)
    ax.scatter([mejor.parametros], [mejor.AUC_validacion], s=140, facecolor="none", edgecolor="#1a1a19", lw=1.5, label="elegida")
    for _, r in fuera.iterrows():
        ax.scatter([r.parametros], [piso - 0.0003], marker="v", s=30, color=AZUL if r.optimizador == "adam" else NARANJA)
    if len(fuera):
        texto = "; ".join(f"config {r.config} ({r.optimizador.upper()}): AUC {r.AUC_validacion:.3f}" for _, r in fuera.iterrows())
        ax.text(0.99, 0.04, "▼ fuera de la escala: " + texto, transform=ax.transAxes, ha="right", fontsize=6.5, color=GRIS)
    ax.set_ylim(piso - 0.001, tabla.AUC_validacion.max() + 0.002)
    ax.set_xscale("log"); ax.set_xlabel("Parámetros entrenables (escala log)"); ax.set_ylabel("AUC en validación")
    ax.set_title(f"Búsqueda aleatoria: {len(tabla)} configuraciones"); ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ruta, dpi=160); plt.close(fig)


def busqueda(cfg, X, y, sal, demo):
    n = cfg["demo"]["n_configuraciones"] if demo else cfg["busqueda"]["n_configuraciones"]
    max_ep = cfg["demo"]["max_epocas"] if demo else cfg["max_epocas"]
    rng = np.random.default_rng(cfg["semilla"])
    filas, t0 = [], time.time()
    for i in range(n):
        hp = sortear_configuracion(rng, cfg["busqueda"])
        modelo, hist, seg = entrenar(X["entrenamiento"], y["entrenamiento"], X["validacion"], y["validacion"],
                                     hp, cfg["semilla"], max_ep, cfg["paciencia"])
        p_va = ModeloRed(modelo, hp).predict_proba(X["validacion"])[:, 1]
        m = evaluar(y["validacion"], p_va)
        filas.append({"config": i + 1, **{k: str(v) if k == "capas" else v for k, v in hp.items()},
                      "parametros": contar_parametros(modelo), "epocas": len(hist), "segundos": round(seg, 1),
                      "AUC_validacion": round(m["AUC"], 5), "Brier_validacion": round(m["Brier"], 5)})
        print(f"  {i + 1:2d}/{n} capas={hp['capas']} {hp['activacion']:10s} drop={hp['dropout']:.1f} "
              f"l2={hp['l2']:.1e} {hp['optimizador']:4s} lr={hp['lr']:.1e} batch={hp['batch']:4d} -> "
              f"AUC val {m['AUC']:.4f} ({len(hist)} épocas, {seg:.0f} s)")
    tabla = pd.DataFrame(filas).sort_values("AUC_validacion", ascending=False)
    tabla.to_csv(sal["tablas"] / ("busqueda_red_demo.csv" if demo else "busqueda_red.csv"), index=False)
    mejor = tabla.iloc[0]
    hp = {"capas": json.loads(mejor["capas"]), "activacion": mejor["activacion"], "dropout": float(mejor["dropout"]),
          "l2": float(mejor["l2"]), "optimizador": mejor["optimizador"], "lr": float(mejor["lr"]), "batch": int(mejor["batch"])}
    print(f"\nMejor configuración (config {mejor['config']}): {hp}  AUC val {mejor['AUC_validacion']:.4f}")
    print(f"Búsqueda: {n} configuraciones en {time.time() - t0:.0f} s")
    if not demo:
        grafica_busqueda(tabla, sal["figuras"] / "fig06_busqueda_red.png")
    return hp, n, time.time() - t0


def semillas(cfg, hp, X, y, sal, demo):
    n = cfg["demo"]["n_semillas"] if demo else cfg["final"]["n_semillas"]
    max_ep = cfg["demo"]["max_epocas"] if demo else cfg["max_epocas"]
    filas, mejor, t0 = [], None, time.time()
    for k in range(n):
        semilla = cfg["semilla"] + 1000 + k
        modelo, hist, seg = entrenar(X["entrenamiento"], y["entrenamiento"], X["validacion"], y["validacion"],
                                     hp, semilla, max_ep, cfg["paciencia"])
        envoltura = ModeloRed(modelo, hp)
        fila = {"semilla": semilla, "epocas": len(hist), "segundos": round(seg, 1)}
        for parte in ("validacion", "prueba"):
            m = evaluar(y[parte], envoltura.predict_proba(X[parte])[:, 1])
            fila.update({f"{k2}_{parte}": m[k2] for k2 in METRICAS})
        filas.append(fila)
        print(f"  semilla {semilla}: AUC val {fila['AUC_validacion']:.4f}  AUC prueba {fila['AUC_prueba']:.4f}  "
              f"({len(hist)} épocas, {seg:.0f} s)")
        if mejor is None or fila["AUC_validacion"] > mejor[0]:
            mejor = (fila["AUC_validacion"], semilla, envoltura, hist, contar_parametros(modelo))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(sal["tablas"] / ("semillas_red_demo.csv" if demo else "semillas_red.csv"), index=False, float_format="%.5f")
    return tabla, mejor, time.time() - t0


def resumen(tabla, columnas):
    return {c: {"media": tabla[c].mean(), "desv": tabla[c].std(ddof=1), "mejor": tabla[c].max() if not c.startswith("Brier") else tabla[c].min(),
                "peor": tabla[c].min() if not c.startswith("Brier") else tabla[c].max()} for c in columnas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pd_red.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    ap.add_argument("--etapa", choices=["todo", "busqueda", "semillas"], default="todo")
    ap.add_argument("--demo", action="store_true", help="muestra del 15 %% de los datos, menos de 2 minutos")
    args = ap.parse_args()
    cfg = aplicar_cambios(leer_yaml(args.config), args.set)
    cfg_pd = leer_yaml("config/pd.yaml")
    sal = {k: RAIZ / v for k, v in cfg_pd["salidas"].items()}
    for d in sal.values():
        d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    demo = cfg["demo"]["fraccion"] if args.demo else None

    partes, prep, X, y, numericas, categoricas = preparar(cfg_pd, demo)
    print(f"{X['entrenamiento'].shape[0]} créditos de entrenamiento, {X['entrenamiento'].shape[1]} entradas"
          + ("  [MODO DEMO]" if demo else ""))

    ruta_busqueda = sal["tablas"] / "busqueda_red_resultado.json"
    if args.etapa in ("todo", "busqueda"):
        hp, n_conf, seg_busq = busqueda(cfg, X, y, sal, demo)
        if not demo:
            json.dump({"hp": hp, "configuraciones": n_conf, "segundos": round(seg_busq, 1)},
                      open(ruta_busqueda, "w", encoding="utf-8"), indent=2)
        if args.etapa == "busqueda":
            return
    else:
        guardado = json.load(open(ruta_busqueda, encoding="utf-8"))
        hp, n_conf, seg_busq = guardado["hp"], guardado["configuraciones"], guardado["segundos"]
    if cfg["final"]["hp"]:
        hp = cfg["final"]["hp"]

    print(f"\nEntrenando la configuración final con varias semillas: {hp}")
    tabla, mejor, seg_sem = semillas(cfg, hp, X, y, sal, demo)
    cols = [f"{m}_{p}" for p in ("validacion", "prueba") for m in METRICAS]
    res = resumen(tabla, cols)
    print("\n" + pd.DataFrame(res).T.round(4).to_string())
    if demo:
        return

    _, semilla, envoltura, hist, n_param = mejor
    sc = cfg_pd["scorecard"]
    p_tr = envoltura.predict_proba(X["entrenamiento"])[:, 1]
    tr = partes["entrenamiento"]
    log = joblib.load(sal["modelos"] / "pd_logistica.joblib")
    joblib.dump({**{k: log[k] for k in ("perfil", "rangos", "opciones", "tasa_por_grado")},
                 "preprocesador": prep, "modelo": envoltura, "entradas": numericas + categoricas,
                 "nombres": list(prep.get_feature_names_out()), "numericas": numericas, "categoricas": categoricas,
                 "scorecard": sc, "puntajes_entrenamiento": np.sort(puntaje_desde_prob(p_tr, sc["puntaje_base"], sc["odds_base"], sc["pdo"])).astype(np.float32),
                 "nombre_modelo": f"Red neuronal {hp['capas']}", "hp": hp, "semilla": semilla},
                sal["modelos"] / "pd_red.joblib")

    json.dump({"hp": hp, "parametros_entrenables": n_param, "configuraciones_busqueda": n_conf,
               "segundos_busqueda": round(seg_busq, 1), "segundos_semillas": round(seg_sem, 1),
               "n_semillas": len(tabla), "semilla_desplegada": int(semilla),
               "resumen": {k: {kk: round(float(vv), 5) for kk, vv in v.items()} for k, v in res.items()}},
              open(sal["tablas"] / "metricas_pd_red.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    ml = json.load(open(sal["tablas"] / "metricas_pd_logistica.json", encoding="utf-8"))
    filas = []
    for parte in ("validacion", "prueba"):
        for m in METRICAS:
            v = ml["metricas"][parte][m]
            filas.append({"conjunto": parte, "metrica": m, "modelo": "Regresión logística", "media": v, "desv": 0.0, "mejor": v, "peor": v})
            r = res[f"{m}_{parte}"]
            filas.append({"conjunto": parte, "metrica": m, "modelo": "Red neuronal", **r})
    comp = pd.DataFrame(filas)
    comp.to_csv(sal["tablas"] / "comparacion_pd.csv", index=False, float_format="%.4f")
    costo = pd.DataFrame([
        {"modelo": "Regresión logística", "parametros": ml["parametros_entrenables"], "configuraciones": ml["configuraciones_probadas"],
         "segundos_totales": ml["segundos_entrenamiento"]},
        {"modelo": "Red neuronal", "parametros": n_param, "configuraciones": n_conf, "segundos_totales": round(seg_busq + seg_sem, 1)}])
    costo.to_csv(sal["tablas"] / "costo_pd.csv", index=False)
    print("\n" + costo.to_string(index=False))

    log_m = log["modelo"]
    p_log = log_m.predict_proba(log["preprocesador"].transform(partes["prueba"][log["entradas"]]))[:, 1]
    p_red = envoltura.predict_proba(X["prueba"])[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for p, nombre, color, ls in [(p_log, "Logística", AZUL, "--"), (p_red, "Red neuronal", NARANJA, "-")]:
        pr, ob = curva_calibracion(y["prueba"], p)
        axes[0].plot(pr * 100, ob * 100, color=color, ls=ls, marker="o", ms=4, lw=2, label=nombre)
    lim = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, lim], [0, lim], color=GRIS, lw=0.8, ls=(0, (2, 2)), label="perfecta")
    axes[0].set_xlabel("Probabilidad predicha (%)"); axes[0].set_ylabel("Tasa observada (%)")
    axes[0].set_title("Calibración en prueba (2015)"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].boxplot([tabla["AUC_prueba"]], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor="#f6c7b2", edgecolor=NARANJA), medianprops=dict(color="#1a1a19"))
    axes[1].scatter(np.random.default_rng(0).normal(1, 0.04, len(tabla)), tabla["AUC_prueba"], s=12, color=NARANJA, zorder=3)
    axes[1].axhline(ml["metricas"]["prueba"]["AUC"], color=AZUL, ls="--", lw=1.5, label="Logística")
    axes[1].set_xticks([1], ["Red neuronal\n(30 semillas)"]); axes[1].set_ylabel("AUC en prueba")
    axes[1].set_title("Variabilidad entre semillas"); axes[1].legend(frameon=False, fontsize=7)
    h = np.array(hist)
    axes[2].plot(h[:, 0], h[:, 1], color=AZUL, lw=2, label="entrenamiento")
    axes[2].plot(h[:, 0], h[:, 2], color=NARANJA, lw=2, ls="--", label="validación")
    axes[2].axvline(h[np.argmin(h[:, 2]), 0], color=GRIS, lw=0.8, ls=(0, (2, 2)), label="parada temprana")
    axes[2].set_xlabel("Época"); axes[2].set_ylabel("Pérdida (entropía cruzada)")
    axes[2].set_title(f"Curva de aprendizaje (semilla {semilla})"); axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig07_red_vs_logistica.png", dpi=160); plt.close(fig)
    print(f"\nModelo final en {sal['modelos'] / 'pd_red.joblib'} (semilla {semilla}, la mejor en validación)")


if __name__ == "__main__":
    main()
