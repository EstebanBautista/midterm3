"""Interfaz Streamlit del clasificador de polaridad.

Ejecución desde la raíz del repositorio:
    .venv/Scripts/streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modelo import DEFAULT_VERSION, cargar_modelo, predecir  # noqa: E402
from src.monitoreo import registrar  # noqa: E402

COLORES = {"negativo": "#e4572e", "neutral": "#9aa0a6", "positivo": "#29a35a"}
ORDEN = ["negativo", "neutral", "positivo"]

st.set_page_config(page_title="Evaluación docente · Polaridad", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Cargando modelo…")
def modelo():
    return cargar_modelo(DEFAULT_VERSION)


_, meta = modelo()

with st.sidebar:
    st.header("Modelo")
    st.markdown(f"**{meta['modelo']}** + {meta['representacion']}  \nVersión `{meta['version']}`")
    c1, c2 = st.columns(2)
    c1.metric("macro-F1 test", f"{meta['macro_f1_test']:.3f}")
    c2.metric("Accuracy test", f"{meta['accuracy_test']:.3f}")
    st.caption(f"Entrenado con {meta['n_train']} comentarios · scikit-learn {meta['scikit_learn']}")
    st.divider()
    st.caption("Herramienta de apoyo: los casos de baja confianza requieren revisión manual. Los comentarios no deben contener datos personales.")

st.title("🎓 Clasificador de comentarios de evaluación docente")
st.write("Clasifica comentarios de estudiantes en **negativo**, **neutral** o **positivo**.")

tab1, tab2 = st.tabs(["✍️ Comentario individual", "📄 Archivo CSV"])

with tab1:
    texto = st.text_area("Comentario", height=140,
                         placeholder="Ej.: El profesor explica muy bien, pero los parciales fueron demasiado difíciles.")
    if st.button("Clasificar", type="primary"):
        r = predecir([texto])[0]
        registrar([r])
        if not r["valido"]:
            st.warning(r["error"])
        else:
            col1, col2 = st.columns([1, 2])
            col1.markdown(f"### <span style='color:{COLORES[r['polaridad']]}'>{r['polaridad'].upper()}</span>",
                          unsafe_allow_html=True)
            col1.metric("Confianza", f"{100 * r['confianza']:.1f} %")
            if r["confianza"] < 0.5:
                col1.info("Confianza baja: conviene revisión humana.")
            probs = pd.DataFrame({"probabilidad": r["probabilidades"]}).loc[ORDEN]
            col2.bar_chart(probs, horizontal=True, color="#4c78a8")

with tab2:
    archivo = st.file_uploader("Suba un CSV con una columna de comentarios", type=["csv"])
    if archivo is not None:
        try:
            df = pd.read_csv(archivo, encoding="utf-8-sig")
        except UnicodeDecodeError:
            archivo.seek(0)
            df = pd.read_csv(archivo, encoding="latin-1")
        cols = list(df.columns)
        col = st.selectbox("Columna con el texto", cols, index=cols.index("feedback") if "feedback" in cols else 0)
        if st.button("Clasificar archivo", type="primary"):
            with st.spinner(f"Clasificando {len(df)} comentarios…"):
                res = predecir(df[col].tolist())
                registrar(res)
            df["polaridad_predicha"] = [r["polaridad"] for r in res]
            df["confianza"] = [r["confianza"] for r in res]
            n_inv = sum(not r["valido"] for r in res)
            if n_inv:
                st.warning(f"{n_inv} comentarios vacíos o inválidos no se clasificaron.")

            dist = df["polaridad_predicha"].value_counts().reindex(ORDEN, fill_value=0)
            m = st.columns(3)
            for i, c in enumerate(ORDEN):
                m[i].metric(c.capitalize(), int(dist[c]), f"{100 * dist[c] / max(dist.sum(), 1):.0f} %", delta_color="off")

            g1, g2 = st.columns(2)
            g1.subheader("Distribución")
            g1.bar_chart(dist, color="#4c78a8")
            otro = [c for c in ["curso", "semestre"] if c in df.columns]
            if otro:
                g2.subheader(f"Por {otro[0]}")
                g2.bar_chart(pd.crosstab(df[otro[0]], df["polaridad_predicha"]).reindex(columns=ORDEN, fill_value=0),
                             color=[COLORES[c] for c in ORDEN])
            st.subheader("Resultados")
            st.dataframe(df, use_container_width=True, height=360)
            st.download_button("⬇️ Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               "comentarios_clasificados.csv", "text/csv")
