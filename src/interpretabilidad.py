import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comun import RAIZ, cargar_datos, leer_yaml, particionar
from prediccion import agrupar

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#6b6b66"

ETIQUETAS = {
    "int_rate": "Tasa de interés", "grade": "Grado", "term": "Plazo", "dti": "DTI", "annual_inc": "Ingreso anual",
    "revol_util": "Uso del cupo rotativo", "inq_last_6mths": "Consultas 6 meses", "delinq_2yrs": "Moras 2 años",
    "pub_rec": "Registros públicos", "home_ownership": "Vivienda", "purpose": "Propósito",
    "antiguedad_credito": "Años de historia", "emp_length": "Antigüedad laboral", "verification_status": "Verificación ingreso",
    "loan_amnt": "Monto", "installment": "Cuota", "open_acc": "Cuentas abiertas", "total_acc": "Cuentas totales",
    "revol_bal": "Saldo rotativo", "tot_cur_bal": "Saldo total", "total_rev_hi_lim": "Cupo rotativo total",
    "tot_coll_amt": "Monto en cobranza", "mths_since_last_delinq": "Meses desde última mora",
    "mths_since_last_record": "Meses desde último registro", "mths_since_last_major_derog": "Meses desde última derog.",
    "collections_12_mths_ex_med": "Cobranzas 12 meses", "acc_now_delinq": "Cuentas en mora hoy",
    "addr_state": "Estado", "initial_list_status": "Lista inicial"}

HIPOTESIS = [
    ("H1", "int_rate", "q"), ("H1", "grade", "c"), ("H2", "term", "c"), ("H3", "dti", "q"), ("H4", "annual_inc", "q"),
    ("H5", "revol_util", "q"), ("H6", "inq_last_6mths", [-1, 0, 1, 2, 3, 100]), ("H7", "delinq_2yrs", [-1, 0, 1, 100]),
    ("H7", "pub_rec", [-1, 0, 1, 100]), ("H8", "home_ownership", "c"), ("H9", "purpose", "c"),
    ("H10", "antiguedad_credito", "q"), ("H11", "emp_length", "c"), ("H12", "verification_status", "c")]


def importancia_permutacion(paquete, df, y, repeticiones, rng):
    base = roc_auc_score(y, paquete["modelo"].predict_proba(paquete["preprocesador"].transform(df[paquete["entradas"]]))[:, 1])
    filas = []
    for v in paquete["entradas"]:
        caidas = []
        for _ in range(repeticiones):
            d = df[paquete["entradas"]].copy()
            d[v] = rng.permutation(d[v].values)
            p = paquete["modelo"].predict_proba(paquete["preprocesador"].transform(d))[:, 1]
            caidas.append(base - roc_auc_score(y, p))
        filas.append({"variable": v, "caida_auc": np.mean(caidas), "desv": np.std(caidas, ddof=1)})
    return pd.DataFrame(filas), base


def formato(v):
    return f"{v / 1000:.0f} mil" if abs(v) >= 10000 else f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


def grupos(df, v, cortes):
    if cortes == "q":
        g = pd.qcut(df[v], 5, duplicates="drop")
        nombres = [f"{formato(max(iv.left, df[v].min()))}–{formato(iv.right)}" for iv in g.cat.categories]
        return g.cat.rename_categories(nombres)
    if cortes == "c":
        return df[v].astype(str)
    g = pd.cut(df[v], cortes)
    return g.cat.rename_categories(["0", "1", "2", "3", "4+"] if len(cortes) == 6 else ["0", "1", "2+"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=4000)
    ap.add_argument("--repeticiones", type=int, default=5)
    args = ap.parse_args()
    cfg = leer_yaml("config/pd.yaml")
    sal = {k: RAIZ / v for k, v in cfg["salidas"].items()}
    df, _, _ = cargar_datos(cfg)
    prueba = particionar(df, cfg)["prueba"].reset_index(drop=True)
    y = prueba["malo"].values
    rng = np.random.default_rng(cfg["semilla"])
    paquetes = {"Regresión logística": joblib.load(sal["modelos"] / "pd_logistica.joblib"),
                "Red neuronal": joblib.load(sal["modelos"] / "pd_red.joblib")}

    tablas = []
    for nombre, paq in paquetes.items():
        t, base = importancia_permutacion(paq, prueba, y, args.repeticiones, rng)
        t.insert(0, "modelo", nombre)
        tablas.append(t)
        print(f"{nombre}: AUC prueba {base:.4f}")
    imp = pd.concat(tablas)
    imp.to_csv(sal["tablas"] / "importancia_permutacion.csv", index=False, float_format="%.5f")
    orden = imp[imp.modelo == "Red neuronal"].sort_values("caida_auc", ascending=False)["variable"].head(15).tolist()[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pos = np.arange(len(orden))
    for k, (nombre, color) in enumerate([("Regresión logística", AZUL), ("Red neuronal", NARANJA)]):
        t = imp[imp.modelo == nombre].set_index("variable").loc[orden]
        ax.barh(pos + (k - 0.5) * 0.38, t.caida_auc * 100, height=0.38, xerr=t.desv * 100, color=color, label=nombre,
                error_kw=dict(lw=0.6, ecolor=GRIS))
    ax.set_yticks(pos, [ETIQUETAS.get(v, v) for v in orden], fontsize=7)
    ax.set_xlabel("Caída del AUC al permutar la variable (puntos porcentuales)")
    ax.set_title("Importancia por permutación en prueba (2015)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(sal["figuras"] / "fig08_importancia_permutacion.png", dpi=160); plt.close(fig)

    red = paquetes["Red neuronal"]
    muestra = prueba.sample(min(args.muestra, len(prueba)), random_state=cfg["semilla"]).reset_index(drop=True)
    X = red["preprocesador"].transform(muestra[red["entradas"]])
    x0 = red["preprocesador"].transform(pd.DataFrame([red["perfil"]])[red["entradas"]])[0]
    factor = red["scorecard"]["pdo"] / np.log(2)
    ig = red["modelo"].gradientes_integrados(X, x0)
    puntos = agrupar(red, -factor * ig)
    completitud = np.abs(ig.sum(axis=1) - (red["modelo"].logit(X) - red["modelo"].logit(x0[None, :])[0])).max()
    print(f"Gradientes integrados en {len(muestra)} créditos; error máximo de completitud {completitud:.4f} (en logit)")
    puntos.abs().mean().sort_values(ascending=False).rename("puntos_abs_medios").to_csv(
        sal["tablas"] / "red_puntos_gradientes_integrados.csv", float_format="%.3f")

    filas = []
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    for ax, (h, v, cortes) in zip(axes.ravel(), HIPOTESIS):
        g = grupos(muestra, v, cortes)
        t = pd.DataFrame({"grupo": g, "puntos": puntos[v]}).groupby("grupo", observed=True)["puntos"].agg(["mean", "size"])
        if v == "emp_length":
            orden_emp = ["< 1 year"] + [f"{i} year" + ("s" if i > 1 else "") for i in range(1, 10)] + ["10+ years"]
            t = t.loc[[o for o in orden_emp if o in t.index]]
        elif v == "purpose":
            t = t[t["size"] >= 30].sort_values("mean")
        for grupo, r in t.iterrows():
            filas.append({"hipotesis": h, "variable": v, "grupo": str(grupo), "puntos_medios": r["mean"], "n": int(r["size"])})
        colores = [NARANJA if m < 0 else AZUL for m in t["mean"]]
        horizontal = v == "purpose"
        if horizontal:
            ax.barh(t.index.astype(str), t["mean"], color=colores, height=0.6); ax.axvline(0, color=GRIS, lw=0.8)
            ax.tick_params(axis="y", labelsize=6)
        else:
            ax.bar(range(len(t)), t["mean"], color=colores, width=0.6); ax.axhline(0, color=GRIS, lw=0.8)
            ax.set_xticks(range(len(t)), t.index.astype(str), rotation=35, ha="right", fontsize=6)
        ax.set_title(f"{h} · {ETIQUETAS[v]}", fontsize=8); ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(HIPOTESIS):]:
        ax.axis("off")
    fig.text(0.5, 0.005, "Puntos medios del scorecard de la red (gradientes integrados) frente al cliente típico. "
             "Naranja: resta puntos (más riesgo). Azul: suma puntos.", ha="center", fontsize=7, color=GRIS)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(sal["figuras"] / "fig09_hipotesis_red.png", dpi=160); plt.close(fig)
    pd.DataFrame(filas).to_csv(sal["tablas"] / "hipotesis_puntos_red.csv", index=False, float_format="%.2f")
    print(f"Figuras fig08 y fig09 en {sal['figuras']}")


if __name__ == "__main__":
    main()
