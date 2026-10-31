# Registro de uso de IA

Vamos anotando aquí los prompts importantes (literales) y qué hicimos con la respuesta.
Esto alimenta la sección de uso de IA y la cacería de la alucinación del reporte.

| Fecha | Quién | Herramienta | Prompt (literal o resumen) | Qué usamos / qué cambiamos |
|---|---|---|---|---|
| 2026-10-26 | Diego | Claude | Explicar el enunciado y armar la estructura del repositorio | Estructura de carpetas, README, requirements |
| 2026-10-26 | Diego | Claude | Script de limpieza y variable objetivo (paso 1) | `src/preparar_datos.py`, `config/datos.yaml`; revisamos a mano la lista de variables |
| 2026-10-26 | Diego | Claude | Análisis descriptivo e hipótesis (paso 2) | `src/descriptivo.py`, Figuras 1–4, `docs/03_hipotesis.md` |
| 2026-10-27 | Diego | Claude | Particiones, regresión logística y scorecard | `src/comun.py`, `src/pd_logistica.py`, `docs/04_modelo_referencia.md` |
| 2026-10-27 | Diego | Claude | Aplicación web con Streamlit | `app/app.py`, `src/prediccion.py` |
| 2026-10-27 | Diego | Claude | Red neuronal para la PD, búsqueda aleatoria, 30 semillas, interpretabilidad | `src/red.py`, `src/pd_red.py`, `src/interpretabilidad.py`, `docs/05_red_neuronal.md` |
| 2026-10-30 | Diego (para que Lenny lo revise y ejecute) | Claude | "sigamos con la parte dos, dame los archivos del punto dos para pasarselos a lenny" | `src/lgd_objetivo.py`, `src/lgd_modelos.py`, `src/perdida_esperada.py`, `config/lgd.yaml`, `docs/06_lgd.md` |

## Verificaciones hechas (sirven para la cacería si no salen 5 alucinaciones)

- El diccionario (`LCDataDictionary.xlsx`) no tiene `verification_status`: la columna aparece como
  `is_inc_v`. Tampoco trae `verification_status_joint` (aparece como `verified_status_joint`).
  Y al revés: el diccionario describe `fico_range_low/high`, que no vienen en `loan.csv`.
  Lo comprobamos comparando las columnas del csv contra la hoja LoanStats.
- La descripción de `loan_status` en el diccionario es solo "Current status of the loan": no
  define las categorías. Su codificación la tomamos de la tabla del enunciado.
- Las definiciones de `funded_amnt`, `total_rec_prncp`, `recoveries` y `collection_recovery_fee`
  usadas en la LGD se copiaron del diccionario, no de la IA.
- Las referencias de `docs/06_lgd.md` (Papke y Wooldridge 1996, Ferrari y Cribari-Neto 2004,
  BCBS 2006) se buscaron en internet para confirmar que existen. Falta revisar las de `docs/05`.

## Posibles alucinaciones (para revisar con evidencia)

Errores reales que salieron durante el desarrollo. Falta completar el prompt literal de cada uno.

- **Código (desactualizado):** la IA usó el parámetro `use_container_width=True` en los elementos de la app.
  Streamlit avisa que ese parámetro está obsoleto y que hay que usar `width="stretch"`.
  Evidencia: el aviso de deprecación al correr la app.
- **Código:** el preprocesamiento usaba una función `lambda` dentro de `FunctionTransformer`, y al
  guardar el modelo con `joblib` falló porque las lambdas no se pueden serializar con pickle.
  Corrección: funciones definidas en el módulo (`es_faltante`, `nombres_nunca` en `src/comun.py`).
- **Código:** en la LGD la IA definió la clase `LinealRecortado` dentro del script principal; al
  cargar el modelo desde otro script (`perdida_esperada.py`) falló con
  `AttributeError: Can't get attribute 'LinealRecortado' on <module '__main__'>`. Corrección:
  mover la clase a `src/comun.py` y reentrenar.
- **Código (desactualizado):** `plt.boxplot(..., vert=True)` da `MatplotlibDeprecationWarning` en
  Matplotlib 3.11. Se quitó el parámetro.
