# Registro de uso de IA

Vamos anotando aquí los prompts importantes (literales) y qué hicimos con la respuesta.
Esto alimenta la sección de uso de IA y la cacería de la alucinación del reporte.

| Fecha | Quién | Herramienta | Prompt (literal o resumen) | Qué usamos / qué cambiamos |
|---|---|---|---|---|
| 2026-10-26 | Diego | Claude | Explicar el enunciado y armar la estructura del repositorio | Estructura de carpetas, README, requirements |
| 2026-10-26 | Diego | Claude | Script de limpieza y variable objetivo (paso 1) | `src/preparar_datos.py`, `config/datos.yaml`; revisamos a mano la lista de variables |

## Verificaciones hechas (sirven para la cacería si no salen 5 alucinaciones)

- El diccionario (`LCDataDictionary.xlsx`) no tiene `verification_status`: la columna aparece como
  `is_inc_v`. Tampoco trae `verification_status_joint` (aparece como `verified_status_joint`).
  Y al revés: el diccionario describe `fico_range_low/high`, que no vienen en `loan.csv`.
  Lo comprobamos comparando las columnas del csv contra la hoja LoanStats.
- La descripción de `loan_status` en el diccionario es solo "Current status of the loan": no
  define las categorías. Su codificación la tomamos de la tabla del enunciado.

## Posibles alucinaciones (para revisar con evidencia)

- 
