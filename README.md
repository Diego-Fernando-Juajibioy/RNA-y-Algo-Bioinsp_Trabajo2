# Trabajo 2 – Redes neuronales para datos tabulares (¿me prestas?)

Redes neuronales y algoritmos bioinspirados · UNAL Medellín · 2026-02

Integrantes: Lenny Alejandro Carvajal Taborda, Diego Fernando Juajibioy Chavez

Queremos responder dos preguntas que se hace un banco antes de prestar:

1. ¿Qué tan probable es que esta persona no pague? → **PD**, clasificación binaria.
2. Si no paga, ¿qué fracción del crédito se pierde? → **LGD**, regresión.

y juntarlas en la pérdida esperada E[L] = PD · LGD · EAD.

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

## Paso 1: preparar los datos

```bash
python src/preparar_datos.py
```

Lee `data/raw/loan.csv`, construye la variable objetivo `malo` (tabla del enunciado), deja solo
las variables que se conocen el día de la solicitud y guarda:

- `data/processed/creditos_pd.csv.gz`: lo que usan los modelos.
- `data/processed/creditos_post.csv.gz`: pagos y recuperaciones, solo para construir la LGD.
- `docs/02_variables.md`: cada variable con su descripción del diccionario oficial.
- `reports/resumen_datos.txt`: conteos para el reporte.

Los parámetros están en `config/datos.yaml`.

## Paso 2: análisis descriptivo

```bash
python src/descriptivo.py
```

Tasa de malos por año, por categoría y por quintil de cada variable (Figuras 1 a 4 en
`reports/figures/`) y la tabla con los números en `reports/tablas/`. Las hipótesis y lo que
muestran las figuras están en `docs/03_hipotesis.md`.

## Paso 3: particiones, regresión logística y scorecard

```bash
python src/pd_logistica.py
python src/pd_logistica.py --set scorecard.pdo=40
```

Particiones temporales (entrenamiento sep-2012 a jun-2014, validación jul-dic 2014, prueba 2015),
preprocesamiento común, regresión logística como modelo de referencia (con búsqueda de C en
validación) y su scorecard. Guarda el modelo en `models/pd_logistica.joblib`, las métricas en
`reports/tablas/` y la Figura 5. Detalles en `docs/04_modelo_referencia.md`.

## Paso 4: red neuronal para la PD

```bash
python src/pd_red.py --demo                 # prueba rápida, menos de 1 minuto
python src/pd_red.py                        # experimento completo
python src/pd_red.py --etapa busqueda       # solo la búsqueda de hiperparámetros
python src/pd_red.py --etapa semillas       # solo las 30 semillas (usa la búsqueda guardada)
python src/pd_red.py --set max_epocas=30 --set final.n_semillas=10
```

Perceptrón multicapa en PyTorch. Búsqueda aleatoria de 30 configuraciones (capas, neuronas,
activación, dropout, L2, optimizador, tasa de aprendizaje, tamaño de lote) elegidas por AUC en
validación, y luego 30 semillas de la mejor. Guarda `models/pd_red.joblib`, las tablas
`busqueda_red.csv`, `semillas_red.csv`, `comparacion_pd.csv` y `costo_pd.csv`, y las Figuras 6 y 7.
Parámetros en `config/pd_red.yaml`. La teoría y los resultados están en `docs/05_red_neuronal.md`.

El experimento completo tarda unos 20 minutos en un portátil con 2 núcleos; por eso existe `--demo`.

## Paso 5: interpretabilidad

```bash
python src/interpretabilidad.py
```

Importancia por permutación de cada variable para la logística y la red (Figura 8), y puntos del
scorecard de la red con gradientes integrados agrupados por cada variable de las hipótesis
(Figura 9). Necesita que existan los dos modelos de los pasos 3 y 4.

## Aplicación web

```bash
streamlit run app/app.py
```

Usa el modelo indicado en `config/app.yaml` (por defecto la red, `pd_red.joblib`; se puede
cambiar a `pd_logistica.joblib`). Se abre en el navegador (http://localhost:8501). La persona
ingresa sus datos y ve su puntaje, su probabilidad de incumplir, su percentil frente a los
clientes de entrenamiento, cómo cambia el puntaje si cambia una característica y qué le suma o
le resta. Los enlaces al reporte y al video se ponen en `config/app.yaml`.

## Estructura

```
config/            parámetros de cada experimento (yaml)
data/raw/          archivo original de Kaggle (no versionado)
data/processed/    datos limpios que usan los modelos
src/               código
models/            modelos entrenados
reports/figures/   figuras del reporte
reports/tablas/    tablas con los resultados
app/               aplicación web
docs/              notas, uso de IA y cacería de alucinaciones
```
