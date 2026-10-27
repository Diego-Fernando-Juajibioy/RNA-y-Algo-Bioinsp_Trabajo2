# 5. Red neuronal para la PD

Se corre con:

```
python src/pd_red.py --demo      # prueba rápida (15 % de los datos, menos de 1 minuto)
python src/pd_red.py             # búsqueda de 30 configuraciones + 30 semillas
python src/interpretabilidad.py  # importancia por permutación y confirmación de hipótesis
```

Se puede correr por partes (`--etapa busqueda` y luego `--etapa semillas`) y cambiar cualquier
valor de `config/pd_red.yaml` sin tocar el código, por ejemplo `--set max_epocas=30`.

Usamos **exactamente las mismas particiones y el mismo preprocesamiento** que la regresión
logística (paso 4), para que la comparación sea justa.

## 5.1 Fundamento matemático

### La neurona y la red

Una neurona recibe un vector $\mathbf x$, hace una combinación lineal y le aplica una función de
activación $g$:

$$a = g(\mathbf w^\top \mathbf x + b)$$

Un perceptrón multicapa (MLP) apila $L$ capas de neuronas. Con $\mathbf h^{(0)} = \mathbf x$:

$$\mathbf h^{(l)} = g\!\left(W^{(l)} \mathbf h^{(l-1)} + \mathbf b^{(l)}\right), \qquad l = 1,\dots,L$$

y la salida es una sola neurona sin activación (el *logit*):

$$z = \mathbf w^{(L+1)\top} \mathbf h^{(L)} + b^{(L+1)}, \qquad \hat p = \sigma(z) = \frac{1}{1+e^{-z}}$$

Si quitamos todas las capas ocultas queda $\hat p = \sigma(\mathbf w^\top\mathbf x + b)$, que es justo
la regresión logística. Por eso la logística es el punto de comparación natural: la red es una
logística a la que se le permite construir sus propias variables ($\mathbf h^{(L)}$) en vez de usar
las originales.

Activaciones que probamos:

| Activación | $g(u)$ | $g'(u)$ |
|---|---|---|
| ReLU | $\max(0,u)$ | $1$ si $u>0$, $0$ si no |
| Leaky ReLU | $\max(0{,}01u,\,u)$ | $1$ si $u>0$, $0{,}01$ si no |
| tanh | $\tanh(u)$ | $1-\tanh^2(u)$ |

Sin activación no lineal la red entera sería una función lineal (componer funciones lineales da
otra lineal) y no ganaríamos nada frente a la logística.

### Función de pérdida: entropía cruzada binaria

Suponemos que $y_i \mid \mathbf x_i \sim \text{Bernoulli}(\hat p_i)$. La verosimilitud de la muestra es

$$\mathcal L(\theta) = \prod_{i=1}^n \hat p_i^{\,y_i} (1-\hat p_i)^{1-y_i}$$

Maximizarla es lo mismo que minimizar menos el logaritmo promedio:

$$J(\theta) = -\frac1n \sum_{i=1}^n \Big[y_i \ln \hat p_i + (1-y_i)\ln(1-\hat p_i)\Big]$$

Esa es la entropía cruzada binaria (BCE). En el código usamos `BCEWithLogitsLoss`, que recibe $z$ en
vez de $\hat p$ y calcula lo mismo de forma numéricamente estable (no se calcula $\ln 0$).

Una cuenta útil: la derivada respecto al logit es simplemente

$$\frac{\partial J_i}{\partial z_i} = \hat p_i - y_i$$

### Retropropagación

Para entrenar necesitamos $\partial J / \partial W^{(l)}$ en todas las capas. Llamando
$\mathbf u^{(l)} = W^{(l)}\mathbf h^{(l-1)} + \mathbf b^{(l)}$ y
$\boldsymbol\delta^{(l)} = \partial J / \partial \mathbf u^{(l)}$, la regla de la cadena da:

$$\delta^{(L+1)} = \hat p - y, \qquad
\boldsymbol\delta^{(l)} = \left(W^{(l+1)\top} \boldsymbol\delta^{(l+1)}\right) \odot g'\!\left(\mathbf u^{(l)}\right)$$

$$\frac{\partial J}{\partial W^{(l)}} = \boldsymbol\delta^{(l)} \, \mathbf h^{(l-1)\top}, \qquad
\frac{\partial J}{\partial \mathbf b^{(l)}} = \boldsymbol\delta^{(l)}$$

Es decir: el error se calcula en la salida y se "devuelve" capa por capa multiplicando por los pesos
y por la derivada de la activación. PyTorch hace esto automáticamente (`perdida.backward()`).

### Optimizadores

Trabajamos con mini-lotes $B$ de tamaño 512, 1024 o 2048 y gradiente $\mathbf g_t = \nabla_\theta J_B(\theta_t)$.

**SGD con momento** ($\mu = 0{,}9$):

$$\mathbf v_t = \mu\,\mathbf v_{t-1} + \mathbf g_t, \qquad \theta_{t+1} = \theta_t - \eta\,\mathbf v_t$$

**Adam** (Kingma y Ba, 2015) lleva promedios del gradiente y de su cuadrado y escala el paso de cada
parámetro por separado:

$$\mathbf m_t = \beta_1 \mathbf m_{t-1} + (1-\beta_1)\mathbf g_t, \qquad
\mathbf v_t = \beta_2 \mathbf v_{t-1} + (1-\beta_2)\mathbf g_t^2$$

$$\hat{\mathbf m}_t = \frac{\mathbf m_t}{1-\beta_1^t},\quad \hat{\mathbf v}_t = \frac{\mathbf v_t}{1-\beta_2^t},\qquad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon}$$

con $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon=10^{-8}$ (valores por defecto de PyTorch).

### Regularización

Con 135 mil créditos y hasta 160 mil parámetros, la red puede memorizar el entrenamiento. Usamos tres
frenos:

1. **Penalización L2** (`weight_decay`): se minimiza $J(\theta) + \frac{\lambda}{2}\|\theta\|^2$, lo que
   agrega $\lambda\theta$ al gradiente y empuja los pesos hacia cero. Es lo mismo que hace el
   parámetro $C = 1/\lambda$ en la logística.
2. **Dropout** (Srivastava et al., 2014): en cada paso de entrenamiento cada neurona oculta se apaga
   con probabilidad $q$ (y las demás se multiplican por $1/(1-q)$). Obliga a la red a no depender de
   una sola neurona. Al predecir se usan todas.
3. **Parada temprana**: después de cada época se calcula la BCE en validación; si no mejora durante
   6 épocas seguidas se detiene y **se recuperan los pesos de la mejor época**.

### Métricas

- **AUC**: probabilidad de que un malo elegido al azar tenga mayor $\hat p$ que un bueno elegido al
  azar. 0,5 es azar, 1 es perfecto.
- **Gini** $= 2\,\text{AUC} - 1$ (la forma en que suelen reportarlo los bancos).
- **KS** $= \max_s |F_{\text{malos}}(s) - F_{\text{buenos}}(s)|$, la máxima separación entre las
  distribuciones acumuladas de ambos grupos.
- **Brier** $= \frac1n\sum(\hat p_i - y_i)^2$: mide si las probabilidades son correctas en nivel
  (calibración), no solo si ordenan bien.

### Scorecard con la red

En la logística el logit es una suma, $z = \beta_0 + \sum_j \beta_j x_j$, y por eso los puntos de cada
variable salen directo de los coeficientes (paso 4). En la red el logit no es aditivo, así que
usamos **gradientes integrados** (Sundararajan, Taly y Yan, 2017). Tomando como referencia el cliente
típico $\mathbf x^0$ (medianas y modas del entrenamiento):

$$\text{IG}_j(\mathbf x) = (x_j - x^0_j)\int_0^1 \frac{\partial z\big(\mathbf x^0 + \alpha(\mathbf x - \mathbf x^0)\big)}{\partial x_j}\, d\alpha$$

La integral se aproxima con 64 puntos (regla del punto medio). La propiedad clave es la
**completitud**:

$$\sum_j \text{IG}_j(\mathbf x) = z(\mathbf x) - z(\mathbf x^0)$$

Como $\text{Puntaje} = \text{Offset} - \text{Factor}\cdot z$, el puntaje se reparte exactamente en

$$\text{Puntaje}(\mathbf x) = \text{Puntaje}(\mathbf x^0) + \sum_j \underbrace{\left(-\text{Factor}\cdot \text{IG}_j(\mathbf x)\right)}_{\text{puntos de la variable } j}$$

Si el modelo fuera la logística, $\text{IG}_j = \beta_j (x_j - x_j^0)$ y se recupera la tabla del
paso 4. Las columnas *one-hot* de una misma variable se suman para dar un solo valor por variable.
Esto es lo que muestra la gráfica "¿Qué te suma y qué te resta?" de la aplicación.

## 5.2 Búsqueda de hiperparámetros

Búsqueda **aleatoria** (Bergstra y Bengio, 2012) de 30 configuraciones sobre:

| Hiperparámetro | Valores |
|---|---|
| Capas ocultas | 1, 2 o 3 (cada capa con la mitad de neuronas que la anterior) |
| Neuronas en la primera capa | 32, 64, 128, 256 |
| Activación | ReLU, tanh, Leaky ReLU |
| Dropout | 0; 0,1; 0,2; 0,3; 0,5 |
| L2 | log-uniforme entre $10^{-6}$ y $10^{-2}$ |
| Optimizador | Adam (tasa $10^{-4}$ a $3\cdot10^{-3}$) o SGD con momento (tasa $10^{-3}$ a $5\cdot10^{-2}$) |
| Tamaño de lote | 512, 1024, 2048 |

Máximo 60 épocas con paciencia 6. Se elige la configuración con mayor AUC en **validación**
(la prueba no se usa para nada de esto).

Resultados en `reports/tablas/busqueda_red.csv` y Figura 6. Tardó 560 s (9 min) en un portátil con
2 núcleos.

- La configuración elegida (config 21) fue **3 capas ocultas de 256, 128 y 64 neuronas, ReLU,
  dropout 0,3, L2 = 4,1·10⁻⁵, Adam con tasa 1,35·10⁻³ y lotes de 1024**: 68 609 parámetros y AUC de
  validación 0,7257.
- Lo que más llama la atención es lo **plano** que es el resultado: 29 de las 30 configuraciones quedan
  entre 0,716 y 0,726, sin importar si tienen 3 mil o 68 mil parámetros. La logística da 0,7189 en
  validación, así que la red mejora menos de un punto de AUC.
- Adam (azul en la Figura 6) queda en general por encima de SGD con las mismas épocas.
- La config 7 (SGD, tasa baja, dropout 0,5 y lotes de 2048) se quedó en AUC 0,546: casi no alcanzó a
  aprender antes de que la parada temprana la detuviera. Es un buen ejemplo de que una mala
  combinación de tasa de aprendizaje y regularización puede dañar el entrenamiento.

**Figura 6.** AUC en validación de las 30 configuraciones frente a su número de parámetros.
Fuente: elaboración propia con datos de Lending Club (Kaggle).

## 5.3 Resultados con 30 semillas

La red da resultados distintos según la semilla (pesos iniciales, orden de los lotes y dropout).
Por eso la configuración elegida se entrena 30 veces y reportamos media, desviación estándar, mejor
y peor. La logística es determinística, así que tiene un solo valor.

Tablas `reports/tablas/semillas_red.csv`, `comparacion_pd.csv` y `costo_pd.csv`; Figura 7.

| Conjunto | Métrica | Logística | Red: media ± desv. | Red: mejor | Red: peor |
|---|---|---|---|---|---|
| Validación | AUC | 0,7189 | 0,7242 ± 0,0010 | 0,7266 | 0,7228 |
| Validación | Gini | 0,4377 | 0,4484 ± 0,0020 | 0,4531 | 0,4455 |
| Validación | KS | 0,3223 | 0,3316 ± 0,0025 | 0,3364 | 0,3263 |
| Validación | Brier | 0,1757 | 0,1738 ± 0,0002 | 0,1733 | 0,1743 |
| Prueba | AUC | 0,6999 | 0,7068 ± 0,0016 | 0,7099 | 0,7038 |
| Prueba | Gini | 0,3997 | 0,4137 ± 0,0033 | 0,4197 | 0,4076 |
| Prueba | KS | 0,2929 | 0,3037 ± 0,0029 | 0,3095 | 0,2978 |
| Prueba | Brier | 0,1724 | 0,1713 ± 0,0005 | 0,1704 | 0,1727 |

(En el Brier "mejor" es el más bajo.)

| Modelo | Parámetros | Configuraciones probadas | Tiempo total |
|---|---|---|---|
| Regresión logística | 107 | 5 | 9 s |
| Red neuronal | 68 609 | 30 + 30 semillas | 1 206 s (20 min) |

Lo que concluimos:

1. **La red es mejor, pero por poco.** En prueba gana 0,7 puntos de AUC (1,4 de Gini, 1,1 de KS) en
   promedio, y **las 30 semillas** le ganan a la logística (la peor semilla da 0,7038 > 0,6999). O sea
   que la diferencia no es suerte de una semilla.
2. **El costo es mucho mayor:** 640 veces más parámetros y unas 130 veces más tiempo de cómputo, y un
   modelo que no se puede leer directamente. Para un banco, la logística sigue siendo una opción muy
   razonable; la ganancia de la red tendría que traducirse en dinero (menos pérdidas) para
   justificarla.
3. **La variabilidad entre semillas es pequeña** (desviación de 0,0016 en AUC de prueba), menor que la
   diferencia con la logística.
4. **No hay sobreajuste fuerte:** el modelo desplegado (semilla 3042) da AUC 0,726 en entrenamiento y
   0,727 en validación. La parada temprana se activó entre las épocas 9 y 20 (13,5 en promedio).
   En la curva de aprendizaje (Figura 7, derecha) la pérdida de entrenamiento sigue bajando mientras la
   de validación ya no mejora: ahí es donde se detiene.
5. **Calibración:** las dos quedan cerca de la diagonal en prueba (Figura 7, izquierda). La red predice
   en promedio 25,9 % y se observa 25,5 %; la logística predice 24,0 %. Ojo: como la parada temprana se
   hace con la validación (27,1 % de malos), la red termina un poco por encima de la tasa real en
   entrenamiento (24,5 % frente a 22,5 %). Es otra consecuencia del sesgo de selección del paso 2.

El modelo que queda guardado para la aplicación es el de la semilla con mejor AUC en validación
(3042). Su puntaje se calcula con la misma escala que la logística (600 puntos = odds 20:1, PDO 20).

**Figura 7.** Izquierda: calibración por deciles en prueba. Centro: AUC en prueba de las 30 semillas
frente a la logística. Derecha: curva de aprendizaje del modelo desplegado. Fuente: elaboración propia
con datos de Lending Club (Kaggle).

## 5.4 Interpretabilidad y confirmación de hipótesis

Usamos dos herramientas que se complementan (`python src/interpretabilidad.py`, unos 50 s):

- **Importancia por permutación** (Breiman, 2001): se desordena al azar una variable en el conjunto
  de prueba, se vuelve a predecir y se mide cuánto cae el AUC. Si cae mucho, el modelo depende de esa
  variable. Se repite 5 veces y se hace para los dos modelos (Figura 8). Dice **cuánto** importa una
  variable, pero no **en qué dirección**.
- **Gradientes integrados** (sección 5.1) sobre 4 000 créditos de prueba: los puntos que cada variable
  suma o resta frente al cliente típico, promediados por grupo (Figura 9). Esto sí da la dirección.
  El error de completitud con 64 pasos fue de máximo 0,016 en logit (menos de medio punto del puntaje).

### Qué variables importan (Figura 8)

Las dos más importantes en los dos modelos son el **ingreso anual** y la **tasa de interés**, seguidas
de DTI, número total de cuentas, uso del cupo rotativo y monto. La logística depende más de la tasa
de interés (3,3 puntos de AUC) y la red reparte más entre ingreso, tasa y uso del cupo. El grado
aporta poco cuando ya está la tasa de interés (dicen casi lo mismo, ver paso 2).

### Confirmación de hipótesis

Comparamos lo que vimos en el descriptivo (cada variable sola, paso 2) con lo que hace la red cuando
ya tiene todas las variables:

| # | Hipótesis | Descriptivo | Red neuronal (Figura 9) | Conclusión |
|---|---|---|---|---|
| H1 | Peor grado y mayor tasa = más riesgo | Se apoya | Tasa: de +16 puntos (5–10 %) a −9 (18–29 %). Grado: A suma, C a G restan. Segunda variable más importante | **Se confirma** |
| H2 | 60 meses más riesgoso que 36 | Se apoya | 60 meses resta 12 puntos en promedio | **Se confirma** |
| H3 | Más DTI = más riesgo | Se apoya | De +7 a −6 puntos, monótono. Tercera más importante | **Se confirma** |
| H4 | Menos ingreso = más riesgo | Se apoya | De −13 (menos de 42 mil) a +10 (más de 100 mil). La más importante | **Se confirma** |
| H5 | Más uso del cupo = más riesgo | Se apoya | De +8 a −3 puntos | **Se confirma** |
| H6 | Más consultas = más riesgo | Se apoya | De +2 (ninguna) a −3 (4 o más); efecto pequeño | Se confirma, débil |
| H7 | Moras y registros públicos = más riesgo | Débil | Moras: −4 (una) y −8 (dos o más). Registros públicos: casi 0 | Se confirma para moras; **no** para registros públicos |
| H8 | Arriendo más riesgoso que hipoteca | Débil | Arriendo resta 5 puntos frente a hipoteca | **Se confirma** |
| H9 | Negocio propio es lo más riesgoso | Se apoya | *small_business* resta 14 puntos, el que más | **Se confirma** |
| H10 | Más historia crediticia = menos riesgo | Muy débil | De −1 a +5 puntos; aparece más clara que en el descriptivo | Se confirma, débil |
| H11 | Más antigüedad laboral = menos riesgo | No se apoya | Sin patrón (sube y baja entre −2,5 y +1) | **No se confirma** |
| H12 | Ingreso verificado = menos riesgo | Al revés | "No verificado" suma 2 puntos; las demás ~0. Importancia por permutación ≈ 0 | **Se rechaza**; con las demás variables casi no aporta |

Algunas cosas para discutir:

- **H7, registros públicos:** sola, la variable sí se asociaba con más malos; dentro del modelo ya no
  resta. Puede ser porque esa información ya está en `mths_since_last_record` (meses desde el último
  registro), que sí tiene peso en la red.
- **H7, moras en la logística:** en la aplicación con la logística vimos que marcar una mora pasada
  subía un poquito el puntaje, lo que no tiene sentido de negocio. La red sí la castiga (−4 y −8
  puntos). Es un ejemplo de algo que la red capta mejor, posiblemente por la interacción con
  `mths_since_last_delinq` y el indicador `nunca_last_delinq`.
- **H12:** sin las demás variables, verificado parecía más riesgoso; con ellas el efecto casi
  desaparece. Esto apoya la idea del paso 2: "verificado" marcaba a un grupo que ya era más riesgoso
  por otras razones (montos más altos, peor grado), no que la verificación en sí cause riesgo.
- Los puntos de la Figura 9 son **promedios por grupo** de un modelo no lineal: el efecto de una
  variable puede cambiar según el resto del perfil. Para un cliente concreto, la aplicación muestra sus
  propios puntos.

**Figura 8.** Caída del AUC en prueba al permutar cada variable (15 más importantes según la red).
**Figura 9.** Puntos medios del scorecard de la red por grupo de cada variable de las hipótesis.
Fuente de ambas: elaboración propia con datos de Lending Club (Kaggle).

## Referencias

- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Kingma, D. P. y Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958.
- Sundararajan, M., Taly, A. y Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32 (importancia por permutación).
