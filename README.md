# Trabajo 2 – Redes neuronales para datos tabulares (¿me prestas?)

Redes neuronales y algoritmos bioinspirados · UNAL Medellín · 2026-02

Integrantes: Lenny Alejandro Carvajal Taborda, Diego Fernando Juajibioy Chavez

Queremos responder dos preguntas que se hace un banco antes de prestar:

1. ¿Qué tan probable es que esta persona no pague? → **PD**, clasificación binaria.
2. Si no paga, ¿qué fracción del crédito se pierde? → **LGD**, regresión.

y juntarlas en la pérdida esperada E[L] = PD · LGD · EAD.

> Estado: en construcción. Vamos llenando esto a medida que avanzamos.

## Datos

Credit Risk Analysis (Lending Club) de Kaggle:
https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

1. Descargar el zip desde Kaggle (hay que iniciar sesión).
2. Descomprimir y copiar `loan.csv` y `LCDataDictionary.xlsx` en `data/raw/`.

`loan.csv` no se sube al repo porque pesa demasiado; lo que sí se sube es la versión
procesada en `data/processed/` y el código que la genera.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura

```
config/            parámetros de cada experimento (yaml)
data/raw/          archivo original de Kaggle (no versionado)
data/processed/    datos limpios que usan los modelos
src/               código
models/            modelos entrenados
reports/figures/   figuras del reporte
app/               aplicación web
docs/              notas, uso de IA y cacería de alucinaciones
```
