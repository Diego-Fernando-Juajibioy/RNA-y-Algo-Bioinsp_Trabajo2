from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tablas"

AZUL = "#2a78d6"
GRIS = "#6b6b66"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.titlesize": 9})


def tasa_por_grupo(df, columna, grupos):
    t = df.groupby(grupos, observed=True)["malo"].agg(n="size", tasa="mean").reset_index()
    t.columns = ["grupo", "n", "tasa"]
    t["grupo"] = t["grupo"].astype(str)
    t.insert(0, "variable", columna)
    return t


def barras(ax, t, titulo, global_, horizontal=True):
    if horizontal:
        ax.barh(t["grupo"], t["tasa"] * 100, color=AZUL, height=0.6)
        ax.axvline(global_ * 100, color=GRIS, ls=(0, (2, 2)), lw=0.9)
        ax.set_xlabel("% de malos")
        for y, (tasa, n) in enumerate(zip(t["tasa"], t["n"])):
            ax.text(tasa * 100 + 0.4, y, f"{tasa:.0%}  (n={n:,})".replace(",", "."), va="center", fontsize=6.5, color="#3d3d3a")
        ax.set_xlim(0, t["tasa"].max() * 100 * 1.45)
    else:
        ax.bar(t["grupo"], t["tasa"] * 100, color=AZUL, width=0.6)
        ax.axhline(global_ * 100, color=GRIS, ls=(0, (2, 2)), lw=0.9)
        ax.set_ylabel("% de malos")
        ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
    ax.set_title(titulo)


def main():
    cfg = yaml.safe_load(open(RAIZ / "config" / "datos.yaml", encoding="utf-8"))
    df = pd.read_csv(RAIZ / cfg["salida_modelo"], parse_dates=["issue_d"])
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    g = df["malo"].mean()
    print(f"{len(df)} créditos, tasa global de malos {g:.2%}")
    tablas = []

    anio = df["issue_d"].dt.year
    t = tasa_por_grupo(df, "anio_originacion", anio)
    tablas.append(t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8))
    a1.bar(t["grupo"], t["n"] / 1000, color=AZUL, width=0.6)
    a1.set_ylabel("miles de créditos"); a1.set_title("Créditos con desenlace por año")
    barras(a2, t, "Tasa de malos por año de originación", g, horizontal=False)
    fig.tight_layout(); fig.savefig(FIG / "fig01_tasa_por_anio.png", dpi=160); plt.close(fig)

    cats = [("grade", "Grado asignado por LC"), ("term", "Plazo"), ("home_ownership", "Vivienda"),
            ("verification_status", "Verificación de ingresos"), ("emp_length", "Antigüedad laboral"),
            ("purpose", "Propósito del crédito")]
    orden_emp = ["< 1 year"] + [f"{i} year" + ("s" if i > 1 else "") for i in range(1, 10)] + ["10+ years"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for ax, (c, titulo) in zip(axes.ravel(), cats):
        t = tasa_por_grupo(df, c, df[c])
        if c == "emp_length":
            t = t.set_index("grupo").loc[[o for o in orden_emp if o in set(t.grupo)]].reset_index()
        elif c == "purpose":
            t = t.sort_values("tasa")
        tablas.append(t)
        barras(ax, t, titulo, g)
    fig.text(0.5, 0.005, f"Línea punteada: tasa global de malos ({g:.1%}).", ha="center", fontsize=7, color=GRIS)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(FIG / "fig02_tasa_categoricas.png", dpi=160); plt.close(fig)

    nums = [("int_rate", "Tasa de interés (%)", "q"), ("dti", "Deuda / ingreso (DTI)", "q"),
            ("annual_inc", "Ingreso anual (USD)", "q"), ("revol_util", "Uso del cupo rotativo (%)", "q"),
            ("loan_amnt", "Monto pedido (USD)", "q"), ("antiguedad_credito", "Años de historia crediticia", "q"),
            ("inq_last_6mths", "Consultas en los últimos 6 meses", [-1, 0, 1, 2, 3, 100]),
            ("delinq_2yrs", "Moras de 30+ días en 2 años", [-1, 0, 1, 100]),
            ("pub_rec", "Registros públicos negativos", [-1, 0, 1, 100])]
    etiquetas_conteo = {5: ["0", "1", "2", "3", "4+"], 3: ["0", "1", "2+"]}
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, (c, titulo, cortes) in zip(axes.ravel(), nums):
        if cortes == "q":
            grupos = pd.qcut(df[c], 5)
            lim = lambda iv: (max(iv.left, df[c].min()), iv.right)
            nombres = [f"{a:,.0f}–{b:,.0f}".replace(",", ".") if df[c].max() > 100 else f"{a:.1f}–{b:.1f}"
                       for a, b in map(lim, grupos.cat.categories)]
            grupos = grupos.cat.rename_categories(nombres)
            titulo += " · quintiles"
        else:
            grupos = pd.cut(df[c], cortes)
            grupos = grupos.cat.rename_categories(etiquetas_conteo[len(cortes) - 1])
        t = tasa_por_grupo(df, c, grupos)
        tablas.append(t)
        barras(ax, t, titulo, g, horizontal=False)
    fig.text(0.5, 0.005, f"Línea punteada: tasa global de malos ({g:.1%}).", ha="center", fontsize=7, color=GRIS)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(FIG / "fig03_tasa_numericas.png", dpi=160); plt.close(fig)

    vars_falt = ["tot_cur_bal", "total_rev_hi_lim", "mths_since_last_delinq", "emp_length"]
    falt = df.groupby(anio)[vars_falt].agg(lambda s: s.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(7, 3))
    estilos = ["-", "--", "-.", ":"]
    colores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for v, ls, col in zip(vars_falt, estilos, colores):
        ax.plot(falt.index, falt[v], ls=ls, color=col, lw=2, marker="o", ms=3, label=v)
    ax.set_ylabel("% de faltantes"); ax.set_xlabel("Año de originación")
    ax.set_title("Faltantes por año: las variables de buró no existen antes de mediados de 2012")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig04_faltantes_por_anio.png", dpi=160); plt.close(fig)

    pd.concat(tablas).to_csv(TAB / "tasa_malos_por_variable.csv", index=False, float_format="%.4f")
    print(f"Figuras en {FIG} y tabla en {TAB}")


if __name__ == "__main__":
    main()
