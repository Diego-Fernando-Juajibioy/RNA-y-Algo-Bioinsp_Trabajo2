# 1. Caso de uso

> Borrador. Lo ajustamos cuando veamos los datos (sobre todo la lista de variables disponibles).

## ¿Quién usa el modelo?

El analista de crédito (o el sistema automático de aprobación) de una entidad que presta
dinero a personas naturales, en nuestro caso con datos de Lending Club, una plataforma
estadounidense de préstamos entre personas.

## ¿Para qué decisión?

Para decidir **si aprueba o no una solicitud de crédito de consumo** y, si la aprueba, para
estimar cuánto podría perder:

- **Modelo de PD:** probabilidad de que la persona sea "mala" (incumpla). Con esto se aprueba o
  se rechaza (o se pone una tasa más alta).
- **Modelo de LGD:** si la persona incumple, qué fracción del crédito no se recupera.
- Juntando ambos con la exposición (EAD) se obtiene la pérdida esperada, que sirve para fijar
  provisiones y precio del crédito.

## ¿En qué momento?

**Al momento de la solicitud, antes de desembolsar el dinero.**

Esta es la decisión más importante del trabajo, porque define qué variables podemos usar.

## ¿Con qué información?

Solo con lo que se sabe el día que la persona pide el crédito:

- Lo que la persona declara: ingresos, antigüedad laboral, vivienda, propósito del crédito, monto pedido.
- Su historia en el buró de crédito: deudas, número de cuentas, consultas recientes, morosidades pasadas.
- Las condiciones que el banco le ofrece: plazo, tasa de interés, grado asignado.

**No podemos usar** nada que solo se conoce después de prestar: pagos realizados, saldo pendiente,
recuperaciones, fecha del último pago, etc. Usarlas sería hacer trampa (*fuga de información* o
*data leakage*): el modelo se vería muy bueno en las pruebas pero sería inútil en la vida real,
porque el día de la solicitud esos datos no existen.

> Ojo: para construir la **variable objetivo** de la LGD sí necesitamos esas variables de
> "después" (pagos y recuperaciones). Una cosa es la variable que se predice y otra las variables
> con las que se predice.

## Pendiente para revisar con los datos

- [ ] Hacer la lista de variables del diccionario (`data/raw/LCDataDictionary.xlsx`) y marcar
      cuáles se conocen al momento de la solicitud y cuáles no.
- [ ] Decidir si la tasa de interés (`int_rate`) y el grado (`grade`) entran al modelo. Se conocen al
      momento de la solicitud, pero los asigna el mismo Lending Club según el riesgo que calculó,
      así que son casi una "respuesta" de otro modelo. Hay que discutirlo.
