import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from prediccion import (cargar_modelo, completar, contribuciones, curva_que_pasa_si, percentil,
                        probabilidad, puntaje)

st.set_page_config(page_title="¿Me prestas?", page_icon="💳", layout="wide")

CFG = yaml.safe_load(open(RAIZ / "config" / "app.yaml", encoding="utf-8"))

PROPOSITOS = {"debt_consolidation": "Unificar deudas", "credit_card": "Pagar tarjeta de crédito",
              "home_improvement": "Mejoras de vivienda", "other": "Otro", "major_purchase": "Compra grande",
              "small_business": "Negocio propio", "car": "Carro", "medical": "Gastos médicos",
              "moving": "Mudanza", "vacation": "Vacaciones", "house": "Compra de vivienda",
              "wedding": "Matrimonio", "renewable_energy": "Energía renovable", "educational": "Estudios"}
VIVIENDA = {"MORTGAGE": "Propia con hipoteca", "RENT": "Arriendo", "OWN": "Propia pagada", "OTHER": "Otra"}
VERIFICACION = {"Verified": "Verificados por el banco", "Source Verified": "Fuente verificada",
                "Not Verified": "Sin verificar"}
EMPLEO = {"< 1 year": "Menos de 1 año", "1 year": "1 año", "10+ years": "10 años o más"}
NOMBRES = {"loan_amnt": "Monto pedido", "int_rate": "Tasa de interés", "installment": "Cuota mensual",
           "annual_inc": "Ingreso anual", "dti": "Deuda / ingreso (DTI)", "delinq_2yrs": "Moras en 2 años",
           "inq_last_6mths": "Consultas en 6 meses", "mths_since_last_delinq": "Meses desde la última mora",
           "mths_since_last_record": "Meses desde el último registro público",
           "mths_since_last_major_derog": "Meses desde la última mora grave", "open_acc": "Cuentas abiertas",
           "pub_rec": "Registros públicos negativos", "revol_bal": "Saldo rotativo", "revol_util": "Uso del cupo rotativo",
           "total_acc": "Total de cuentas", "collections_12_mths_ex_med": "Cobros en 12 meses",
           "acc_now_delinq": "Cuentas en mora hoy", "tot_coll_amt": "Total en cobro",
           "tot_cur_bal": "Saldo total de deudas", "total_rev_hi_lim": "Cupo rotativo total",
           "antiguedad_credito": "Años de historia crediticia", "term": "Plazo", "grade": "Grado",
           "emp_length": "Antigüedad laboral", "home_ownership": "Vivienda", "verification_status": "Verificación",
           "purpose": "Propósito", "addr_state": "Estado (EE. UU.)", "initial_list_status": "Tipo de listado"}


@st.cache_resource
def modelo():
    return cargar_modelo(CFG["modelo"])


def banda(p):
    if p < 0.10:
        return "Riesgo bajo", "🟢"
    if p < 0.20:
        return "Riesgo moderado", "🟡"
    if p < 0.35:
        return "Riesgo alto", "🟠"
    return "Riesgo muy alto", "🔴"


def formulario(paq):
    perfil, rangos, op = paq["perfil"], paq["rangos"], paq["opciones"]
    d = {}
    st.sidebar.header("Tus datos")
    st.sidebar.caption("Empiezan en los valores típicos de un cliente. Cámbialos por los tuyos.")

    st.sidebar.subheader("El crédito")
    d["loan_amnt"] = st.sidebar.number_input("Monto que pides (USD)", 500, 35000, int(perfil["loan_amnt"]), step=500)
    d["term"] = st.sidebar.radio("Plazo", ["36 months", "60 months"], horizontal=True,
                                 format_func=lambda t: t.replace("months", "meses"))
    d["purpose"] = st.sidebar.selectbox("¿Para qué es?", op["purpose"], format_func=lambda x: PROPOSITOS.get(x, x))
    grados = sorted(op["grade"])
    d["grade"] = st.sidebar.select_slider("Grado que te asignó el banco (A = mejor)", grados, value=perfil["grade"])
    d["int_rate"] = st.sidebar.slider("Tasa de interés anual (%)", 5.0, 29.0, float(paq["tasa_por_grado"][d["grade"]]), 0.1,
                                      help="Por defecto, la tasa típica de tu grado.")

    st.sidebar.subheader("Tu situación")
    d["annual_inc"] = st.sidebar.number_input("Ingreso anual (USD)", 4000, 1000000, int(perfil["annual_inc"]), step=1000)
    d["home_ownership"] = st.sidebar.selectbox("Vivienda", op["home_ownership"], format_func=lambda x: VIVIENDA.get(x, x))
    orden_emp = ["< 1 year"] + [f"{i} year" + ("s" if i > 1 else "") for i in range(1, 10)] + ["10+ years"]
    d["emp_length"] = st.sidebar.selectbox("Tiempo en tu trabajo actual", orden_emp, index=orden_emp.index(perfil["emp_length"]),
                                           format_func=lambda x: EMPLEO.get(x, x.replace("years", "años")))
    d["verification_status"] = st.sidebar.selectbox("Ingresos", op["verification_status"],
                                                    format_func=lambda x: VERIFICACION.get(x, x))

    st.sidebar.subheader("Tu historial de crédito")
    d["dti"] = st.sidebar.slider("Deudas mensuales / ingreso mensual (%)", 0.0, 40.0, float(perfil["dti"]), 0.5,
                                 help="DTI: cuánto de tu ingreso se va en pagar deudas (sin hipoteca).")
    d["revol_util"] = st.sidebar.slider("Uso de tus tarjetas de crédito (%)", 0.0, 100.0, float(perfil["revol_util"]), 1.0)
    d["inq_last_6mths"] = st.sidebar.number_input("Solicitudes de crédito en los últimos 6 meses", 0, 10, int(perfil["inq_last_6mths"]))
    d["delinq_2yrs"] = st.sidebar.number_input("Veces en mora (30+ días) en los últimos 2 años", 0, 10, int(perfil["delinq_2yrs"]))
    d["pub_rec"] = st.sidebar.number_input("Registros públicos negativos", 0, 5, int(perfil["pub_rec"]))
    d["antiguedad_credito"] = st.sidebar.slider("Años desde tu primer crédito", 1.0, 40.0, float(round(perfil["antiguedad_credito"])), 0.5)

    with st.sidebar.expander("Más datos del buró (opcional)"):
        d["open_acc"] = st.number_input("Cuentas de crédito abiertas", 0, 60, int(perfil["open_acc"]))
        d["total_acc"] = st.number_input("Total de cuentas en tu historial", 1, 100, int(perfil["total_acc"]))
        d["revol_bal"] = st.number_input("Saldo en tarjetas (USD)", 0, 300000, int(perfil["revol_bal"]), step=500)
        d["total_rev_hi_lim"] = st.number_input("Cupo total de tus tarjetas (USD)", 0, 500000, int(perfil["total_rev_hi_lim"]), step=500)
        d["tot_cur_bal"] = st.number_input("Saldo total de todas tus deudas (USD)", 0, 2000000, int(perfil["tot_cur_bal"]), step=1000)
        nunca = st.checkbox("Nunca he estado en mora", value=True)
        d["mths_since_last_delinq"] = np.nan if nunca else st.number_input("Meses desde tu última mora", 0, 200, 24)
    return d


def main():
    paq = modelo()
    datos = formulario(paq)
    fila = completar(paq, datos)
    p = float(probabilidad(paq, [fila])[0])
    s = float(puntaje(paq, [p])[0])
    pct = percentil(paq, s)
    texto_banda, icono = banda(p)

    st.title("¿Me prestas? 💳")
    st.markdown("Descubre cómo te vería un banco antes de pedir un crédito: tu **puntaje**, tu **probabilidad de no "
                "pagar** y qué podrías cambiar para mejorar.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tu puntaje", f"{s:.0f}")
    c2.metric("Probabilidad de incumplir", f"{p:.1%}")
    c3.metric("Percentil", f"{pct:.0f}", help=f"Tu puntaje es mayor que el de {pct:.0f} de cada 100 clientes.")
    c4.metric("Nivel de riesgo", f"{icono} {texto_banda.replace('Riesgo ', '').capitalize()}")
    st.caption(f"Cuota mensual estimada: USD {fila['installment']:,.0f} · Modelo: {paq['nombre_modelo']} · "
               f"Escala: {paq['scorecard']['puntaje_base']} puntos = odds {paq['scorecard']['odds_base']}:1, "
               f"cada {paq['scorecard']['pdo']} puntos las odds se duplican.")

    st.subheader("¿Dónde quedas frente a los demás?")
    pob = paq["puntajes_entrenamiento"]
    hist = pd.DataFrame({"puntaje": pob[:: max(1, len(pob) // 20000)]})
    base = alt.Chart(hist).mark_bar(color="#9ec5f4").encode(
        alt.X("puntaje:Q", bin=alt.Bin(maxbins=50), title="Puntaje"), alt.Y("count()", title="Clientes"))
    linea = alt.Chart(pd.DataFrame({"puntaje": [s], "texto": ["Tú"]})).mark_rule(color="#eb6834", size=3).encode(x="puntaje:Q")
    etiqueta = alt.Chart(pd.DataFrame({"puntaje": [s], "texto": ["Tú"]})).mark_text(
        dy=-110, color="#eb6834", fontWeight="bold").encode(x="puntaje:Q", text="texto")
    st.altair_chart((base + linea + etiqueta).properties(height=240), width="stretch")
    st.caption("Distribución de puntajes de los clientes con los que se entrenó el modelo (Lending Club, 2012–2014).")

    st.subheader("¿Qué pasaría si…?")
    izq, der = st.columns([1, 2])
    opciones_var = ["dti", "revol_util", "annual_inc", "loan_amnt", "int_rate", "inq_last_6mths", "term", "grade", "home_ownership"]
    var = izq.selectbox("Cambia una característica", opciones_var, format_func=lambda v: NOMBRES[v])
    if var in paq["opciones"]:
        valores = sorted(paq["opciones"][var])
    else:
        lo, hi = paq["rangos"][var]
        lo, hi = min(lo, fila[var]), max(hi, fila[var])
        valores = list(np.round(np.linspace(lo, hi, 40), 2))
    puntos = curva_que_pasa_si(paq, fila, var, valores)
    curva = pd.DataFrame({NOMBRES[var]: [str(v) for v in valores] if var in paq["opciones"] else valores, "Puntaje": puntos})
    if var in paq["opciones"]:
        graf = alt.Chart(curva).mark_bar(color="#2a78d6").encode(x=alt.X(NOMBRES[var], sort=None), y=alt.Y("Puntaje", scale=alt.Scale(zero=False)))
    else:
        graf = alt.Chart(curva).mark_line(color="#2a78d6", strokeWidth=3).encode(x=NOMBRES[var], y=alt.Y("Puntaje", scale=alt.Scale(zero=False)))
        actual = alt.Chart(pd.DataFrame({NOMBRES[var]: [fila[var]], "Puntaje": [s]})).mark_point(
            color="#eb6834", size=120, filled=True).encode(x=NOMBRES[var], y="Puntaje")
        graf = graf + actual
    der.altair_chart(graf.properties(height=260), width="stretch")
    izq.caption("La línea muestra tu puntaje si solo cambiaras esa característica y dejaras todo lo demás igual. "
                "El punto naranja eres tú hoy.")

    cambios = {"dti": max(0.0, fila["dti"] - 5), "revol_util": max(0.0, fila["revol_util"] - 20),
               "inq_last_6mths": 0, "term": "36 months", "loan_amnt": max(500, fila["loan_amnt"] * 0.75)}
    filas_ideas = []
    for v, nuevo in cambios.items():
        if fila[v] == nuevo:
            continue
        delta = float(curva_que_pasa_si(paq, fila, v, [nuevo])[0]) - s
        filas_ideas.append({"Si…": {"dti": "bajas tu DTI 5 puntos", "revol_util": "usas 20 % menos de tus tarjetas",
                                    "inq_last_6mths": "no pides más créditos por 6 meses", "term": "pides a 36 meses",
                                    "loan_amnt": "pides 25 % menos"}[v], "tu puntaje cambia": round(delta, 1)})
    if filas_ideas:
        st.markdown("**Ideas para subir tu puntaje**")
        st.dataframe(pd.DataFrame(filas_ideas).sort_values("tu puntaje cambia", ascending=False),
                     hide_index=True, width="content")

    aportes = contribuciones(paq, fila)
    if aportes is not None:
        st.subheader("¿Qué te suma y qué te resta?")
        top = pd.concat([aportes.head(5), aportes.tail(5)]).drop_duplicates()
        top = top[top.abs() > 0.05]
        if len(top):
            df_ap = pd.DataFrame({"Característica": [NOMBRES.get(i, i) for i in top.index], "Puntos": top.values})
            df_ap["Efecto"] = np.where(df_ap["Puntos"] > 0, "Te suma", "Te resta")
            g = alt.Chart(df_ap).mark_bar().encode(
                x=alt.X("Puntos:Q", title="Puntos frente a un cliente típico"),
                y=alt.Y("Característica:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("Efecto:N", scale=alt.Scale(domain=["Te suma", "Te resta"], range=["#2a78d6", "#eb6834"])))
            st.altair_chart(g.properties(height=max(140, 34 * len(df_ap) + 40)), width="stretch")
        else:
            st.write("Tus datos son muy parecidos a los de un cliente típico.")

    st.divider()
    st.markdown(f"📄 [Reporte técnico]({CFG['enlace_reporte']}) · 🎬 [Video]({CFG['enlace_video']}) · "
                f"💻 [Código]({CFG['enlace_repositorio']})")
    st.caption("Proyecto académico del curso Redes Neuronales y Algoritmos Bioinspirados (UNAL Medellín, 2026-2). "
               "Datos: Lending Club vía Kaggle (créditos de consumo en EE. UU., 2012–2015). "
               "No es una evaluación real de crédito.")


main()
