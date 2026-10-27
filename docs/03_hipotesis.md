# 3. Análisis descriptivo e hipótesis

Las figuras salen de `python src/descriptivo.py` (carpeta `reports/figures/`) y los números de
`reports/tablas/tasa_malos_por_variable.csv`. La tasa global de malos es **21,9 %** (58 819 de
268 530 créditos).

> Ojo: aquí solo miramos cada variable **por separado**. Que una variable se vea asociada al riesgo
> no quiere decir que aporte algo cuando ya están las demás (por ejemplo, la tasa de interés y el
> grado dicen casi lo mismo). La confirmación final de cada hipótesis va al final del trabajo,
> con el modelo y el análisis de interpretabilidad (tarea 7).

## Hipótesis

Las planteamos pensando en cómo funciona un crédito: alguien es más riesgoso si debe mucho para lo
que gana, si ya ha tenido problemas pagando o si está buscando plata con urgencia.

| # | Hipótesis | Lo que muestra el descriptivo | Estado preliminar |
|---|---|---|---|
| H1 | A peor grado (A → G) y mayor tasa de interés, más riesgo | Grado A: 7 %, G: 47 %. Tasa de interés: del 8 % de malos en el quintil más bajo al 38 % en el más alto (Figuras 2 y 3) | Se apoya, y es la relación más fuerte |
| H2 | Los créditos a 60 meses son más riesgosos que a 36 | 35 % vs. 18 % de malos (Figura 2) | Se apoya |
| H3 | A mayor deuda frente al ingreso (DTI), más riesgo | 16 % en el quintil más bajo vs. 31 % en el más alto (Figura 3) | Se apoya |
| H4 | A menor ingreso anual, más riesgo | 27 % en el quintil más bajo vs. 17 % en el más alto (Figura 3) | Se apoya |
| H5 | Quien usa más su cupo rotativo es más riesgoso | 15 % → 27 % de menor a mayor uso (Figura 3) | Se apoya |
| H6 | Más consultas recientes (busca crédito en varias partes) = más riesgo | 20 % con 0 consultas vs. 28 % con 4 o más (Figura 3) | Se apoya |
| H7 | Haber tenido moras o registros públicos negativos aumenta el riesgo | Sube, pero poco: 21 % → 27 % (moras) y 22 % → 26 % (registros) (Figura 3) | Se apoya débilmente |
| H8 | Los que pagan arriendo son más riesgosos que los que tienen hipoteca | 24 % vs. 20 % (Figura 2) | Se apoya débilmente |
| H9 | Los créditos para negocio propio son los más riesgosos | *small_business*: 32 %, el más alto de todos los propósitos (Figura 2) | Se apoya |
| H10 | Más años de historia crediticia = menos riesgo | Baja muy poco: 23 % → 21 % (Figura 3) | Efecto muy débil |
| H11 | Más antigüedad laboral = menos riesgo | Prácticamente plano, entre 21 % y 23 % (Figura 2) | **No se apoya** |
| H12 | Si el ingreso fue verificado, el riesgo es menor | **Al revés**: verificados 25 %, no verificados 17 % (Figura 2) | **Se rechaza** (ver nota) |

## Cosas que nos llamaron la atención

1. **Ingresos verificados con más malos (H12).** Uno esperaría lo contrario. Una explicación
   posible es que Lending Club verifica sobre todo a quienes le parecen más riesgosos o piden
   montos más altos, entonces "verificado" está marcando a un grupo que ya era más riesgoso.
   Es una hipótesis nuestra que habría que respaldar con una fuente; por ahora no la afirmamos.

2. **La tasa de malos cambia mucho con el año (Figura 1).** Baja de 26 % en 2007 a 14 % en
   2009–2010 y vuelve a subir a 26 % en 2014–2015. Lo de 2014–2015 creemos que es **sesgo de
   selección**: como sacamos los créditos que siguen activos ("Current"), de esos años solo
   quedan los que ya terminaron, y en poco tiempo terminan sobre todo los que cayeron en mora
   rápido (o los que pagaron por anticipado). Esto importa para la partición temporal: si
   evaluamos en 2015, el conjunto de prueba tiene más malos que el de entrenamiento.

3. **Las variables de buró no existen antes de mediados de 2012 (Figura 4).** `tot_cur_bal`,
   `tot_coll_amt` y `total_rev_hi_lim` faltan en el 100 % de los créditos de 2007–2011. No es que
   la gente no tenga saldo, es que no se registraba. Hay que decidir si usamos solo créditos
   desde 2012, o si dejamos todos e imputamos con un indicador de faltante.

4. **`application_type`** tiene solo 3 créditos conjuntos (JOINT) con desenlace. No aporta nada;
   la vamos a quitar.

5. Los faltantes de `mths_since_last_delinq` (55 %) probablemente no son "datos perdidos" sino
   personas que **nunca** han tenido mora (el diccionario no lo dice explícitamente; lo inferimos
   porque esos créditos tienen menos malos: 21,3 % frente a 22,7 %). Si es así, imputar con la
   media sería un error: hay que tratarlos como una categoría aparte.

## Figuras

- **Figura 1.** Créditos con desenlace y tasa de malos por año de originación. Fuente: elaboración
  propia con datos de Lending Club (Kaggle). `reports/figures/fig01_tasa_por_anio.png`
- **Figura 2.** Tasa de malos por categoría de las variables categóricas. Misma fuente.
  `reports/figures/fig02_tasa_categoricas.png`
- **Figura 3.** Tasa de malos por quintil (o por conteo) de las variables numéricas. Misma fuente.
  `reports/figures/fig03_tasa_numericas.png`
- **Figura 4.** Porcentaje de faltantes por año de originación. Misma fuente.
  `reports/figures/fig04_faltantes_por_anio.png`
