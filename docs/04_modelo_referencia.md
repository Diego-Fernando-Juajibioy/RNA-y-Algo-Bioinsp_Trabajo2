# 4. Particiones, modelo de referencia y scorecard

Se corre con `python src/pd_logistica.py` (unos segundos). Resultados en
`reports/tablas/metricas_pd_logistica.json` y Figura 5.

## Datos que usamos

Solo créditos originados **desde septiembre de 2012**. Antes de esa fecha las variables de buró
(`tot_cur_bal`, `tot_coll_amt`, `total_rev_hi_lim`) no se registraban (Figura 4): en agosto de 2012
todavía faltan en el 83 % de los créditos y en septiembre ya solo en el 1 %. Así evitamos que el
modelo aprenda "si falta el dato, el crédito es viejo".

## Particiones (todas temporales)

| Conjunto | Créditos originados | n | % malos |
|---|---|---|---|
| Entrenamiento | sep-2012 a jun-2014 | 135 329 | 22,5 % |
| Validación | jul-2014 a dic-2014 | 34 988 | 27,1 % |
| Prueba | ene-2015 a dic-2015 | 30 835 | 25,5 % |

¿Por qué temporales y no aleatorias? Porque el banco entrena con créditos del pasado y usa el
modelo con solicitudes futuras. Una partición aleatoria mezclaría meses y daría una evaluación
más optimista de lo que pasaría en la realidad.

La validación sirve para escoger hiperparámetros de la red; la prueba se mira una sola vez al final.

Cuidado al interpretar: en validación y prueba hay más malos por el sesgo de selección explicado en
el paso 2 (de 2014–2015 solo quedan los créditos que ya terminaron).

## Preprocesamiento (el mismo para la logística y para la red)

Se ajusta **solo con entrenamiento** y después se aplica igual a validación y prueba:

1. Numéricas: imputación con la mediana; `log(1 + x)` a las variables de dinero (muy sesgadas);
   recorte en los percentiles 1 y 99; estandarización (media 0, desviación 1).
2. Para `mths_since_last_delinq`, `mths_since_last_record` y `mths_since_last_major_derog` se agrega
   un indicador `nunca_...` = 1 si el valor falta (la persona nunca ha tenido ese evento).
3. Categóricas: *one-hot*; las categorías con menos de 500 créditos se agrupan en "infrecuente".
4. Se quitan `application_type` (3 créditos conjuntos) y `sub_grade` (repite lo de `grade` e `int_rate`).

Resultado: 106 entradas.

## Regresión logística

$$P(\text{malo}=1 \mid \mathbf x) = \sigma(\beta_0 + \boldsymbol\beta^\top \mathbf x), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Con penalización L2. El único hiperparámetro es $C$ (inverso de la fuerza de la penalización); probamos
$C \in \{0{,}001;\ 0{,}01;\ 0{,}1;\ 1;\ 10\}$ y elegimos el de mayor AUC en validación. Las cinco dan
prácticamente lo mismo (AUC entre 0,7186 y 0,7189), así que con 135 mil créditos y 107 parámetros la
penalización casi no importa. Quedó $C = 0{,}01$. En total: 5 configuraciones, ~9 s, 107 parámetros.

| | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Entrenamiento | 0,705 | 0,409 | 0,300 | 0,158 |
| Validación | 0,719 | 0,438 | 0,323 | 0,176 |
| Prueba | 0,700 | 0,400 | 0,294 | 0,172 |

En la curva de calibración (Figura 5, izquierda) el modelo queda algo **por debajo** de la tasa
observada en validación (predice 23,8 % en promedio y se observa 27,1 %). Es consistente con el
cambio en la proporción de malos entre periodos.

## Scorecard

Puntaje en función de las *odds* de ser bueno:

$$\text{Puntaje} = \text{Offset} + \text{Factor}\cdot \ln\!\left(\frac{1-p}{p}\right),\qquad
\text{Factor} = \frac{PDO}{\ln 2},\qquad \text{Offset} = \text{Base} - \text{Factor}\cdot\ln(\text{odds}_{base})$$

Con **Base = 600**, **odds de referencia = 20:1** (un 4,8 % de probabilidad de ser malo) y
**PDO = 20**: cada 20 puntos las odds de ser bueno se duplican. Todo se cambia en `config/pd.yaml`
o en la línea de comandos, por ejemplo `python src/pd_logistica.py --set scorecard.pdo=40`.

Como en la logística $\ln\frac{1-p}{p} = -(\beta_0 + \sum_j \beta_j x_j)$, el puntaje se reparte en
una parte fija más los puntos de cada variable:

$$\text{Puntaje} = \underbrace{\text{Offset} - \text{Factor}\,\beta_0}_{\text{puntos base} \,=\, 550{,}2} \;+\; \sum_j \underbrace{\left(-\text{Factor}\,\beta_j x_j\right)}_{\text{puntos de la variable } j}$$

La tabla `reports/tablas/scorecard_logistica_puntos.csv` muestra, para cada variable, el rango de
puntos que aporta. Las de mayor rango son ingreso anual, tasa de interés, monto, número de
cuentas abiertas y DTI.

**Figura 5.** Izquierda: curva de calibración por deciles en validación y prueba. Derecha:
distribución del puntaje en prueba para buenos y malos. Fuente: elaboración propia con datos de
Lending Club (Kaggle).
