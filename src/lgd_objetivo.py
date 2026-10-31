import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import RAIZ, aplicar_cambios, leer_yaml

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#6b6b66"


def construir(cfg):
    post = pd.read_csv(RAIZ / cfg["datos_post"], parse_dates=["issue_d"])
    post["ultimo_pago"] = pd.to_datetime(post["last_pymnt_d"], format="%b-%Y", errors="coerce")
    cast = post[post["loan_status"].isin(cfg["estados_castigados"])].copy()
    cast["ead_obs"] = cast["funded_amnt"] - cast["total_rec_prncp"]
    cast["recuperado_neto"] = cast["recoveries"] - cast["collection_recovery_fee"]
    cast["lgd_sin_recorte"] = 1 - cast["recuperado_neto"] / cast["ead_obs"]
    cast["lgd"] = cast["lgd_sin_recorte"].clip(0, 1)
    return post, cast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/lgd.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = ap.parse_args()
    cfg = aplicar_cambios(leer_yaml(args.config), args.set)
    fig_dir, tab_dir = RAIZ / cfg["salidas"]["figuras"], RAIZ / cfg["salidas"]["tablas"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    post, cast = construir(cfg)
    lineas = [f"Créditos con desenlace: {len(post)}",
              f"Malos (malo = 1): {int(post.malo.sum())}",
              f"Castigados (Charged Off): {len(cast)}",
              f"  con EAD observada <= 0: {(cast.ead_obs <= 0).sum()}",
              f"  con LGD < 0 antes de recortar (recuperaron más que la EAD): {(cast.lgd_sin_recorte < 0).sum()}",
              f"  sin fecha de último pago: {cast.ultimo_pago.isna().sum()}"]

    trim = cast.dropna(subset=["ultimo_pago"]).groupby(cast["ultimo_pago"].dt.to_period("Q"))
    madurez = pd.DataFrame({"n": trim.size(), "lgd_media": trim["lgd"].mean(),
                            "sin_recuperacion": trim["recoveries"].apply(lambda s: (s == 0).mean())})
    madurez = madurez[madurez["n"] >= 30]
    madurez.to_csv(tab_dir / "lgd_madurez_recuperacion.csv", float_format="%.4f")

    corte = pd.Timestamp(cfg["corte_recuperacion"])
    final = cast[(cast["issue_d"] >= cfg["fecha_inicio"]) & (cast["ultimo_pago"] <= corte) & (cast["ead_obs"] > 0)]
    pd_df = pd.read_csv(RAIZ / cfg["datos_pd"], parse_dates=["issue_d"])
    cols_post = ["id", "loan_status", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
                 "ultimo_pago", "ead_obs", "recuperado_neto", "lgd"]
    lgd = pd_df.drop(columns=["loan_status"], errors="ignore").merge(final[cols_post], on="id", how="inner")
    lgd = lgd.sort_values(["issue_d", "id"])
    lgd.to_csv(RAIZ / cfg["datos"], index=False, compression="gzip")
    lineas += [f"Base para la LGD (originados desde {cfg['fecha_inicio']}, último pago hasta {cfg['corte_recuperacion']}): {len(lgd)}",
               f"  LGD media {lgd.lgd.mean():.4f}, mediana {lgd.lgd.median():.4f}, desviación {lgd.lgd.std():.4f}",
               f"  LGD = 1 (nada recuperado): {(lgd.lgd >= 1).mean():.1%}",
               f"  LGD < 0,5: {(lgd.lgd < 0.5).mean():.1%}",
               f"  EAD observada / monto desembolsado: {(lgd.ead_obs / lgd.funded_amnt).mean():.3f}"]
    texto = "\n".join(lineas)
    print(texto)
    (RAIZ / "reports" / "resumen_lgd.txt").write_text(texto + "\n", encoding="utf-8")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = madurez.index.to_timestamp()
    a1.plot(x, madurez["sin_recuperacion"] * 100, color=NARANJA, lw=2, marker="o", ms=3, label="% sin ninguna recuperación")
    a1.plot(x, madurez["lgd_media"] * 100, color=AZUL, lw=2, ls="--", marker="o", ms=3, label="LGD media (%)")
    a1.axvline(corte, color=GRIS, lw=1, ls=(0, (2, 2)))
    a1.text(corte, 50, " corte", color=GRIS, fontsize=7)
    a1.set_xlabel("Trimestre del último pago (≈ momento del incumplimiento)"); a1.set_ylabel("%")
    a1.set_title("Las recuperaciones tardan: los incumplimientos recientes aún no recuperan", fontsize=8)
    a1.legend(frameon=False, fontsize=7)
    a2.hist(lgd["lgd"], bins=50, color=AZUL, edgecolor="white", lw=0.3)
    a2.set_xlabel("LGD"); a2.set_ylabel("Créditos")
    a2.set_title(f"Distribución de la LGD (n = {len(lgd):,})".replace(",", "."), fontsize=8)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(fig_dir / "fig10_lgd_objetivo.png", dpi=160); plt.close(fig)
    print(f"Datos en {RAIZ / cfg['datos']} y Figura 10 en {fig_dir}")


if __name__ == "__main__":
    main()
