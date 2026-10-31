# 6. Pérdida dado incumplimiento (LGD) y pérdida esperada

Se corre con:

```
python src/lgd_objetivo.py       # construye la variable objetivo y data/processed/creditos_lgd.csv.gz
python src/lgd_modelos.py --demo # prueba rápida (menos de 2 minutos)
python src/lgd_modelos.py        # regresión lineal + búsqueda de 30 redes + 30 semillas (~6 min)
python src/perdida_esperada.py   # E[L] = PD · LGD · EAD en el conjunto de prueba de la PD
```

Todo se configura en `config/lgd.yaml`, por ejemplo `--set red.perdida=mse` o
`--set perdida_esperada.ead=ajustada`.

## 6.1 Variable objetivo

### Fórmula

Para cada crédito castigado:

$$\text{EAD}_{obs} = \texttt{funded\_amnt} - \texttt{total\_rec\_prncp}$$

$$\text{LGD} = 1 - \frac{\texttt{recoveries} - \texttt{collection\_recovery\_fee}}{\text{EAD}_{obs}}, \qquad \text{recortada a } [0, 1]$$

### Por qué cada campo

Las definiciones son las del diccionario oficial (`data/raw/LCDataDictionary.xlsx`):

| Campo | Diccionario | Para qué lo usamos |
|---|---|---|
| `funded_amnt` | "The total amount committed to that loan at that point in time." | Lo que realmente se prestó. Usamos este y no `loan_amnt` (lo que se pidió) porque en algunos créditos se prestó menos de lo pedido. |
| `total_rec_prncp` | "Principal received to date" | Capital que la persona alcanzó a pagar antes de incumplir. Restado del desembolso da el capital que quedó expuesto. |
| `recoveries` | "post charge off gross recovery" | Lo que se recuperó después del castigo, en bruto. |
| `collection_recovery_fee` | "post charge off collection fee" | Lo que costó recuperar (agencia de cobro). Se resta porque no es dinero que le quede al banco. |

### Qué créditos entran

1. Solo los **castigados** (`Charged Off` y `Does not meet the credit policy. Status:Charged Off`).
   Los `Default` y `Late (31-120 days)` cuentan como malos para la PD, pero todavía no tienen
   recuperaciones (las recuperaciones empiezan después del castigo), así que su LGD saldría 1 por
   construcción.
2. Originados desde septiembre de 2012, igual que la PD (variables de buró).
3. Con **último pago hasta diciembre de 2014**. La Figura 10 (izquierda) muestra por qué: a partir de
   2015 el porcentaje de créditos castigados sin ninguna recuperación pasa de ~30 % a más de 90 %.
   No es que esos créditos sean peores; es que los datos se descargaron cuando el proceso de cobro
   apenas empezaba. Sin este corte, la LGD de los años recientes saldría inflada artificialmente.

Quedan **16 253 créditos** (`reports/resumen_lgd.txt`).

### La distribución es rara (Figura 10, derecha)

- Media 0,904, mediana 0,880, desviación 0,093.
- **31,5 % tiene LGD = 1**: no se recuperó nada.
- La mayoría del resto se concentra entre 0,85 y 0,90. Pocos créditos tienen LGD < 0,5 (0,5 %).

Es decir, hay dos "montañas": una en 1 y otra cerca de 0,87. Una posible explicación es que la
recuperación no se negocia caso por caso sino que la deuda castigada se vende o se cobra a un
porcentaje parecido, pero no encontramos una fuente que lo confirme, así que lo dejamos como
hipótesis.

### Limitaciones

- **No incluye intereses ni comisiones.** Una persona pudo haber pagado muchos intereses antes de
  incumplir (`total_rec_int`). Con nuestra definición la pérdida es de capital; si se contaran los
  intereses cobrados como recuperación, la LGD bajaría bastante.
- **No descuenta en el tiempo.** Recuperar 100 dólares dos años después vale menos que 100 hoy.
  No tenemos las fechas de cada recuperación, solo el total.
- **Recuperaciones posiblemente incompletas.** Aun con el corte de 2014, los créditos que
  incumplieron a finales de 2014 tuvieron solo ~12 meses de cobro.
- **38 créditos tienen LGD negativa** antes de recortar (se recuperó más que la EAD, posiblemente
  intereses y cargos incluidos en el recobro). Se recortan a 0.
- **Sesgo de selección en el tiempo.** Dentro de esta base, los créditos más recientes son los que
  incumplieron rápido (si hubieran incumplido tarde, no alcanzarían a estar antes del corte).

## 6.2 Particiones y preprocesamiento

Temporales, por fecha de originación, y las mismas para los dos modelos de LGD:

| Conjunto | Originados | n | LGD media |
|---|---|---|---|
| Entrenamiento | sep-2012 a jun-2013 | 7 698 | 0,903 |
| Validación | jul-2013 a dic-2013 | 4 781 | 0,901 |
| Prueba | ene-2014 a dic-2014 | 3 774 | 0,910 |

Las entradas son **las mismas variables de solicitud que la PD**, porque el caso de uso es el mismo:
estimar al momento de la solicitud cuánto se perdería si la persona incumple. El preprocesamiento es
el mismo del paso 4 (`src/comun.py`), ajustado solo con el entrenamiento de la LGD. La única
diferencia es `min_frecuencia_categoria: 100` en vez de 500, porque aquí hay 7 698 créditos y no
135 mil. Resultado: 85 entradas.

## 6.3 Modelos

### Referencia: regresión lineal con penalización (Ridge)

$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\; \sum_i (y_i - \beta_0 - \mathbf x_i^\top\boldsymbol\beta)^2 + \alpha\|\boldsymbol\beta\|^2$$

Con $\alpha \in \{0{,}01;\ 0{,}1;\ 1;\ 10;\ 100;\ 1000\}$ elegido por RMSE en validación ($\alpha = 100$).
Como la recta puede salirse de $[0,1]$, las predicciones se recortan a ese intervalo.
86 parámetros, 6 configuraciones, 0,03 s.

### Red neuronal: función de pérdida y activación de salida

La LGD es una **proporción**: vive en $[0,1]$ y tiene muchos valores exactamente en 1. Eso define
dos decisiones:

**Activación de salida: sigmoide.** $\hat\mu = \sigma(z) \in (0,1)$, así la predicción nunca se sale
del rango posible. (Con salida lineal habría que recortar, como con la regresión lineal.)

**Pérdida: entropía cruzada binaria (cuasi-verosimilitud de Bernoulli).** Papke y Wooldridge (1996)
proponen, para variables fraccionarias, maximizar

$$\ell(\theta) = \sum_i \Big[y_i \ln \hat\mu_i + (1-y_i)\ln(1-\hat\mu_i)\Big], \qquad y_i \in [0,1]$$

que es la log-verosimilitud de Bernoulli aunque $y_i$ no sea 0 o 1. Funciona porque la derivada de
cada término respecto a $\hat\mu$ es

$$\frac{\partial \ell_i}{\partial \hat\mu_i} = \frac{y_i - \hat\mu_i}{\hat\mu_i(1-\hat\mu_i)}$$

y su valor esperado dado $\mathbf x$ es cero exactamente cuando $\hat\mu = E[y \mid \mathbf x]$. O sea:
el estimador apunta a la **media condicional** de la LGD, que es lo que necesitamos para la pérdida
esperada, sin suponer ninguna distribución para $y$. Además, como en la PD,
$\partial \ell_i / \partial z_i = y_i - \hat\mu_i$, así que reutilizamos todo el código.

**¿Por qué no otras?**

- **MSE:** sale de máxima verosimilitud suponiendo $y \sim \mathcal N(\mu, \sigma^2)$:
  $-\ln L = \frac n2\ln(2\pi\sigma^2) + \frac1{2\sigma^2}\sum (y_i-\mu_i)^2$, y minimizar eso en $\mu$ es
  minimizar $\sum(y_i-\mu_i)^2$. También estima la media condicional, pero la normal supone
  simetría y soporte en todos los reales, lo que no se cumple aquí. Además, con salida sigmoide el
  gradiente de MSE se multiplica por $\sigma'(z)$, que es casi cero cerca de 0 o 1, y la red aprende
  más lento justo en la zona donde están los datos (LGD ≈ 0,9–1). Está disponible con
  `--set red.perdida=mse`.
- **Regresión beta** (Ferrari y Cribari-Neto, 2004): es un modelo natural para proporciones, pero
  la densidad beta no está definida en 0 ni en 1, y el 31,5 % de nuestros datos vale exactamente 1.

El resto (búsqueda aleatoria de 30 configuraciones, 30 semillas, parada temprana, Adam/SGD, L2,
dropout) es igual que en la PD (sección 5), con lotes más pequeños (128, 256, 512), hasta 200 épocas y
paciencia de 15, porque hay muchos menos datos. Se elige por RMSE en validación.

## 6.4 Métricas

- $\text{MAE} = \frac1n\sum|y_i - \hat y_i|$: error promedio en las mismas unidades de la LGD.
- $\text{RMSE} = \sqrt{\frac1n\sum(y_i - \hat y_i)^2}$: castiga más los errores grandes.
- $R^2 = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$: cuánto mejora frente a predecir siempre la
  media del mismo conjunto. Puede ser negativo.
- **Distancia de Wasserstein-1** $W_1 = \int_0^1 |F^{-1}(u) - \hat F^{-1}(u)|\,du$, la diferencia
  promedio entre los cuantiles observados y predichos. La usamos para comparar **distribuciones**,
  como pide el enunciado: un modelo puede tener buen RMSE prediciendo siempre la media, pero su
  distribución sería un solo punto.

También comparamos contra un modelo **trivial**: predecir siempre la LGD media del entrenamiento.

## 6.5 Resultados

Tablas `comparacion_lgd.csv`, `costo_lgd.csv`, `distribucion_lgd_prueba.csv` y
`semillas_lgd_red.csv`; Figuras 11 y 12.

| Conjunto | Métrica | Media de entrenamiento | Regresión lineal | Red: media ± desv. | Red: mejor | Red: peor |
|---|---|---|---|---|---|---|
| Validación | MAE | 0,0703 | 0,0669 | 0,0661 ± 0,0004 | 0,0653 | 0,0667 |
| Validación | RMSE | 0,0916 | 0,0900 | 0,0899 ± 0,0001 | 0,0897 | 0,0901 |
| Validación | R² | −0,000 | 0,033 | 0,036 ± 0,002 | 0,039 | 0,032 |
| Prueba | MAE | 0,0674 | 0,0655 | 0,0654 ± 0,0002 | 0,0651 | 0,0657 |
| Prueba | RMSE | 0,0905 | **0,0898** | 0,0903 ± 0,0001 | 0,0901 | 0,0907 |
| Prueba | R² | −0,007 | **0,009** | −0,002 ± 0,003 | 0,003 | −0,011 |
| Prueba | W1 | 0,0672 | 0,0557 | **0,0542** ± 0,0005 | 0,0532 | 0,0552 |

| Modelo | Parámetros | Configuraciones | Tiempo total |
|---|---|---|---|
| Regresión lineal | 86 | 6 | 0,03 s |
| Red neuronal | 55 041 (capas 256 y 128, ReLU, dropout 0,1, Adam) | 30 + 30 semillas | 349 s |

Distribución en prueba (2014):

| | p5 | p25 | p50 | p75 | p95 | media | desv. |
|---|---|---|---|---|---|---|---|
| Observada | 0,802 | 0,859 | 0,883 | 1,000 | 1,000 | 0,910 | 0,090 |
| Regresión lineal | 0,879 | 0,896 | 0,906 | 0,917 | 0,931 | 0,906 | 0,016 |
| Red neuronal | 0,871 | 0,892 | 0,905 | 0,916 | 0,931 | 0,903 | 0,018 |

**Lo que concluimos:**

1. **Con la información de la solicitud, la LGD casi no se puede predecir.** Los dos modelos explican
   menos del 4 % de la varianza en validación y prácticamente nada en prueba. El MAE baja de 0,067 a
   0,065 frente a predecir siempre la media: menos de dos puntos de LGD.
2. **La red no le gana a la regresión lineal.** En prueba la lineal tiene menor RMSE que las 30
   semillas de la red (la mejor semilla da 0,0901 > 0,0898). En MAE empatan. La red solo gana un
   poco en $W_1$, porque sus predicciones están un poco más dispersas. Con 640 veces más parámetros
   y miles de veces más tiempo de cómputo, la red no se justifica para la LGD.
3. **Las predicciones están muy concentradas** (Figura 12, izquierda): desviación de 0,018 frente a
   0,090 observada. Ningún modelo predice LGD = 1 ni LGD de 0,80. Es lo esperado cuando el modelo
   no tiene información: si no sabe distinguir, lo óptimo para el error cuadrático es predecir cerca
   de la media.
4. **Poca variabilidad entre semillas** (desviación de 0,0001 en RMSE). La red siempre llega casi al
   mismo lugar, porque no hay mucha señal que aprender. La parada temprana se activó entre las épocas
   35 y 88; en la curva de aprendizaje (Figura 12, derecha) la pérdida de validación se aplana muy
   rápido.
5. **Lo que sí importaría para la LGD** probablemente es información posterior al desembolso: cuánto
   alcanzó a pagar, cuándo incumplió, cómo se gestionó el cobro. Pero eso no se conoce el día de la
   solicitud, y nuestro caso de uso no lo permite.

**Figura 10.** Izquierda: porcentaje de créditos castigados sin recuperaciones y LGD media por
trimestre del último pago. Derecha: distribución de la LGD en la base final.
**Figura 11.** RMSE en validación de las 30 configuraciones de la red y de la regresión lineal.
**Figura 12.** Izquierda: distribución observada y predicha de la LGD en prueba. Centro: RMSE en prueba
de las 30 semillas. Derecha: curva de aprendizaje del modelo guardado.
Fuente de todas: elaboración propia con datos de Lending Club (Kaggle).

## 6.6 Pérdida esperada

$$E[L] = PD \cdot LGD \cdot EAD$$

(Comité de Supervisión Bancaria de Basilea, 2006).

**EAD que usamos: el monto pedido (`loan_amnt`).** Es lo único que se conoce con seguridad el día de la
solicitud, y es un límite superior: en el peor caso la persona no paga ninguna cuota. En la base de
la LGD, al momento del castigo quedaba expuesto en promedio el 75 % de lo desembolsado (porque ya
habían pagado parte del capital). Por eso dejamos la opción `--set perdida_esperada.ead=ajustada`, que
multiplica el monto por ese 0,748 aprendido en el entrenamiento de la LGD.

Aplicamos los dos modelos al conjunto de **prueba de la PD** (créditos de 2015, 30 835) y comparamos
con una pérdida "observada" aproximada: para los malos, capital no pagado menos recuperaciones netas;
para los buenos, cero (`perdida_esperada_prueba.csv`, Figura 13).

| Combinación | PD media | LGD media | Pérdida esperada / monto |
|---|---|---|---|
| Red (PD) × Red (LGD) | 0,259 | 0,907 | 25,2 % |
| Logística (PD) × Lineal (LGD) | 0,240 | 0,909 | 23,3 % |
| Observado (aproximado) | 0,255 | — | 24,4 % |

- A nivel de portafolio las dos combinaciones quedan cerca de lo observado (la red un poco por
  encima, la logística un poco por debajo).
- Por deciles de pérdida esperada (Figura 13, izquierda) la curva sigue muy de cerca a la observada:
  del 5 % en el decil más bajo al 46–50 % en el más alto. **Casi todo ese orden viene de la PD**, porque
  la LGD es casi constante.
- Con la EAD "ajustada" la pérdida esperada baja al 18,9 % y queda por debajo de lo observado. La
  razón es el sesgo de selección de 2015: los créditos que ya terminaron en 2015 son sobre todo los
  que incumplieron muy rápido, sin alcanzar a pagar capital, así que su EAD real está cerca del monto
  completo.
- Ojo: la pérdida "observada" de 2015 está sobreestimada, porque esos créditos casi no han tenido
  tiempo de recuperar (Figura 10). Es una comparación aproximada, no una validación exacta.

**Figura 13.** Pérdida esperada frente a observada en prueba (2015), por decil de pérdida esperada y por
grado. Fuente: elaboración propia con datos de Lending Club (Kaggle).

## Referencias

- Basel Committee on Banking Supervision. (2006). *International convergence of capital measurement and capital standards: A revised framework* (Comprehensive version). Bank for International Settlements. https://www.bis.org/publ/bcbs128.htm
- Ferrari, S. L. P., & Cribari-Neto, F. (2004). Beta regression for modelling rates and proportions. *Journal of Applied Statistics, 31*(7), 799–815. https://doi.org/10.1080/0266476042000214501
- Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632.
