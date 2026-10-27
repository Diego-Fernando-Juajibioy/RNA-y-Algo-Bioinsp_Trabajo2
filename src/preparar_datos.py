import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]


def leer_config(ruta):
    with open(RAIZ / ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def a_meses(fecha_texto):
    return pd.to_datetime(fecha_texto, format="%b-%Y", errors="coerce")


def limpiar_emp_length(s):
    return s.str.strip().replace({"n/a": np.nan})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/datos.yaml")
    p.add_argument("--archivo", help="ruta a loan.csv (si no, la del yaml)")
    args = p.parse_args()
    cfg = leer_config(args.config)
    archivo = Path(args.archivo) if args.archivo else RAIZ / cfg["archivo_original"]

    print(f"Leyendo {archivo} ... (tarda unos 20 s)")
    df = pd.read_csv(archivo, low_memory=False)
    n_original, p_original = df.shape
    resumen = [f"Archivo original: {n_original} registros y {p_original} variables",
               f"Fecha de descarga: {cfg['fecha_descarga']}"]

    df["issue_d"] = a_meses(df["issue_d"])
    df["earliest_cr_line"] = a_meses(df["earliest_cr_line"])
    resumen.append(f"Periodo de originación (issue_d): {df.issue_d.min():%Y-%m} a {df.issue_d.max():%Y-%m}")

    conteo = df["loan_status"].value_counts()
    resumen.append("\nloan_status en el archivo original:")
    resumen += [f"  {k}: {v}" for k, v in conteo.items()]

    df["malo"] = df["loan_status"].map(cfg["objetivo"])
    sin_desenlace = df["malo"].isna().sum()
    df = df[df["malo"].notna()].copy()
    df["malo"] = df["malo"].astype(int)
    resumen.append(f"\nSe excluyen {sin_desenlace} créditos sin desenlace (Current, Issued, In Grace Period, Late 16-30)")
    resumen.append(f"Quedan {len(df)} créditos: {df.malo.sum()} malos ({df.malo.mean():.2%}) y {(df.malo == 0).sum()} buenos")

    df["antiguedad_credito"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    df["term"] = df["term"].str.strip()
    df["emp_length"] = limpiar_emp_length(df["emp_length"])
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})

    numericas = cfg["numericas"] + ["antiguedad_credito"]
    categoricas = cfg["categoricas"]
    columnas_modelo = ["id", "issue_d", "malo"] + numericas + categoricas

    salida = RAIZ / cfg["salida_modelo"]
    salida.parent.mkdir(parents=True, exist_ok=True)
    df[columnas_modelo].to_csv(salida, index=False, compression="gzip")
    df[["id", "issue_d", "loan_status", "malo", "loan_amnt"] + cfg["post_desembolso"]].to_csv(
        RAIZ / cfg["salida_post"], index=False, compression="gzip")

    resumen.append(f"\nVariables de entrada: {len(numericas)} numéricas y {len(categoricas)} categóricas "
                   f"({len(numericas) + len(categoricas)} en total)")
    faltantes = df[numericas + categoricas].isna().mean().sort_values(ascending=False)
    resumen.append("Porcentaje de faltantes (las que tienen alguno):")
    resumen += [f"  {k}: {v:.1%}" for k, v in faltantes[faltantes > 0].items()]

    escribir_tabla_variables(cfg, numericas, categoricas, df.columns)

    (RAIZ / "reports").mkdir(exist_ok=True)
    (RAIZ / "reports" / "resumen_datos.txt").write_text("\n".join(resumen), encoding="utf-8")
    print("\n".join(resumen))
    print(f"\nListo: {salida}")


def escribir_tabla_variables(cfg, numericas, categoricas, columnas):
    dic = pd.read_excel(RAIZ / cfg["diccionario"], sheet_name="LoanStats")
    dic.columns = ["variable", "descripcion"]
    dic["variable"] = dic["variable"].astype(str).str.strip()
    desc = dict(zip(dic.variable, dic.descripcion))
    alias = {"verification_status": "is_inc_v", "verification_status_joint": "verified_status_joint"}
    for nombre_csv, nombre_dic in alias.items():
        desc[nombre_csv] = f"{desc[nombre_dic]} *(en el diccionario aparece como `{nombre_dic}`)*"

    filas = ["# 2. Variables\n",
             "Descripciones copiadas de `data/raw/LCDataDictionary.xlsx`, hoja *LoanStats* "
             "(diccionario oficial incluido en el conjunto de Kaggle). Archivo generado por "
             "`src/preparar_datos.py`; no editar a mano.\n",
             "| Variable | Uso | Descripción (diccionario) |", "|---|---|---|"]
    for v in numericas:
        filas.append(f"| `{v}` | entrada (numérica) | {desc.get(v, '— (derivada: años entre earliest_cr_line e issue_d)')} |")
    for v in categoricas:
        filas.append(f"| `{v}` | entrada (categórica) | {desc.get(v, '—')} |")
    for v in cfg["post_desembolso"]:
        filas.append(f"| `{v}` | solo LGD/EAD (posterior al desembolso) | {desc.get(v, '—')} |")
    usadas = set(numericas) | set(categoricas) | set(cfg["post_desembolso"])
    excluidas = [c for c in columnas if c not in usadas and c not in ("malo", "antiguedad_credito")]
    notas = {"id": "identificador (se guarda para unir tablas, no entra al modelo)",
             "issue_d": "fecha de originación: se usa para la partición temporal, no como entrada",
             "loan_status": "se usa para construir la variable objetivo `malo`"}
    for v in excluidas:
        uso = notas.get(v, "excluida")
        filas.append(f"| `{v}` | {uso} | {desc.get(v, '—')} |")
    (RAIZ / "docs").mkdir(exist_ok=True)
    (RAIZ / "docs" / "02_variables.md").write_text("\n".join(filas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
