"""
Dashboard Interactivo Empresarial de ShopFast (EP3 — IE2, IE5, IE9).
Consume en tiempo real la API REST FastAPI, mostrando recomendaciones con interpretabilidad
para toma de decisiones, métricas de calidad del modelo ML y observabilidad RED.
"""

from __future__ import annotations

import os
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_API_KEY = os.getenv("SHOPFAST_API_KEY", "shopfast-ep3-key-2026")

st.set_page_config(
    page_title="ShopFast AI — Panel Empresarial de Recomendaciones",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #0F172A 0%, #1B365D 65%, #2563EB 100%);
            padding: 1.2rem 1.6rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.2rem;
            border-left: 6px solid #FFB81C;
        }
        .main-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; }
        .main-header p { margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
        .rec-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #1B365D;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>ShopFast AI — Panel Ejecutivo de Recomendación e Interpretabilidad</h1>
        <p>EP3 Arquitectura de Sistemas de IA (ITY1102) • Equipo 5: Araceli Salgado &amp; Andrés Concha • SLA Objetivo &lt; 200 ms</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, api_url: str) -> dict | None:
    try:
        r = httpx.get(f"{api_url}{path}", timeout=4.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"No se pudo consultar {path}: {exc}")
        return None


def api_post(path: str, payload: dict, api_url: str, api_key: str) -> dict | None:
    try:
        r = httpx.post(
            f"{api_url}{path}",
            json=payload,
            headers={"X-API-Key": api_key},
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Error en petición POST {path}: {exc}")
        return None


with st.sidebar:
    st.subheader("Conexión API REST")
    api_url = st.text_input("URL Base API", value=API_BASE_URL)
    api_key = st.text_input("X-API-Key", value=DEFAULT_API_KEY, type="password")

    health = api_get("/health", api_url)
    if health:
        st.success(f"API Operativa • CB: {health.get('circuit_breaker')}")
        st.caption(f"Catálogo: {health.get('catalog_items')} productos | Cola: {health.get('queue_backend')}")
    else:
        st.warning("API no disponible en la URL configurada.")

    st.divider()
    st.subheader("Perfil del Cliente (Simulador)")
    user_id = st.text_input("ID de Cliente (anonimizado SHA-256)", value="cliente_1045")
    categoria = st.selectbox(
        "Categoría de Interés",
        ["Laptops", "Smartphones", "Audio", "Gaming", "Monitores", "Almacenamiento", "Redes", "Accesorios"],
        index=0,
    )
    presupuesto = st.slider("Presupuesto Cliente (USD)", min_value=50, max_value=2500, value=950, step=25)
    vistas = st.slider("Vistas previas en categoría", min_value=0, max_value=20, value=5)
    dias = st.slider("Días desde última compra", min_value=0, max_value=180, value=14)
    carrito = st.slider("Ítems en carrito", min_value=0, max_value=5, value=1)
    top_k = st.slider("Top-K recomendaciones", min_value=3, max_value=8, value=5)
    cyberday = st.checkbox("Modo Evento CyberDay", value=True)
    simular_fallo = st.checkbox("Simular fallo / activar Circuit Breaker", value=False)

tab1, tab2, tab3 = st.tabs(
    [
        "1. Recomendaciones e Interpretabilidad (IE2)",
        "2. Calidad del Modelo ML (IE10)",
        "3. Observabilidad RED y Mensajería (IE4/IE7)",
    ]
)

with tab1:
    payload = {
        "user_id": user_id,
        "categoria_preferida": categoria,
        "presupuesto_usd": float(presupuesto),
        "vistas_previas_categoria": int(vistas),
        "dias_desde_ultima_compra": int(dias),
        "items_en_carrito": int(carrito),
        "es_cyberday": bool(cyberday),
        "top_k": int(top_k),
        "simular_fallo": bool(simular_fallo),
    }

    if "last_pred" not in st.session_state or st.button("Generar Recomendaciones en Vivo", type="primary"):
        res = api_post("/predecir", payload, api_url, api_key)
        if res:
            st.session_state["last_pred"] = res

    pred = st.session_state.get("last_pred")
    if pred:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latencia Inferencia", f"{pred['latencia_inferencia_ms']} ms", "SLA < 200 ms")
        c2.metric("Motor / Origen", pred["origen_inferencia"])
        c3.metric("Circuit Breaker", "FALLBACK" if pred["circuit_breaker_activo"] else "NORMAL (CLOSED)")
        c4.metric("ID Anonimizado (Ley 19.628)", pred["user_hash_anonimizado"])

        recs = pred["recomendaciones"]
        df_recs = pd.DataFrame(recs)

        col_left, col_right = st.columns([1.3, 1.0])
        with col_left:
            st.subheader("Top Productos Recomendados + Decisión de Negocio")
            for idx, r in enumerate(recs, start=1):
                reasons_html = " • ".join(r["explicacion_negocio"])
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <strong>#{idx} {r['nombre']}</strong> <code>{r['producto_id']}</code><br/>
                        Categoría: <b>{r['categoria']}</b> | Precio: <b>USD {r['precio_usd']:.2f}</b> ({r['descuento_pct']}% dcto) |
                        Score Afinidad: <b>{r['score_afinidad']:.2%}</b> ({r['nivel_confianza']}) |
                        Estrategia: <b>{r['estrategia_comercial']}</b> (Δ Ticket: {r['impacto_ticket_usd']:+.2f} USD)<br/>
                        <small><i>Por qué se recomienda:</i> {reasons_html}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_right:
            st.subheader("Ranking por Probabilidad de Conversión")
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            names = [n[:28] + "..." if len(n) > 28 else n for n in df_recs["nombre"][::-1]]
            scores = df_recs["score_afinidad"][::-1] * 100.0
            ax.barh(names, scores, color="#1B365D")
            ax.set_xlabel("Score de Afinidad (%)")
            ax.set_xlim(0, 105)
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            avg_price = float(df_recs["precio_usd"].mean())
            st.info(
                f"Ticket promedio sugerido en esta sesión: **USD {avg_price:.2f}** "
                f"(meta negocio ShopFast: elevar de USD 85 a USD 120)."
            )

with tab2:
    info = api_get("/model/info", api_url)
    if info:
        m = info["metrics"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{m['accuracy']:.2%}")
        k2.metric("Precision", f"{m['precision']:.2%}")
        k3.metric("Recall", f"{m['recall']:.2%}")
        k4.metric("F1-Score", f"{m['f1_score']:.2%}")
        k5.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

        st.subheader("Importancia Global de Características (Explicabilidad del Modelo)")
        imp_df = pd.DataFrame(
            [{"Variable": k, "Importancia (%)": round(v * 100.0, 2)} for k, v in info["feature_importances"].items()]
        )
        col_a, col_b = st.columns([1.2, 1.0])
        with col_a:
            fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
            s_df = imp_df.sort_values("Importancia (%)")
            ax2.barh(s_df["Variable"], s_df["Importancia (%)"], color="#2563EB")
            ax2.set_xlabel("Importancia (%)")
            ax2.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        with col_b:
            st.write("**Hiperparámetros del Modelo Entrenado**")
            st.json(info["hyperparameters"])
            st.write("**Matriz de Confusión (Hold-out 20%)**")
            st.json(info["confusion_matrix"])

with tab3:
    metrics = api_get("/metrics", api_url)
    if metrics:
        red = metrics["red_metrics"]
        cache = metrics["cache"]
        resil = metrics["resiliencia"]
        msg = metrics["mensajeria"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Requests Totales", red["total_requests"])
        m2.metric("Latencia p95", f"{red['latency_ms']['p95']} ms", "Meta < 200 ms")
        m3.metric("Tasa Cache Hit", f"{cache['hit_rate_pct']}%", f"{cache['hits']} hits / {cache['misses']} misses")
        m4.metric("Eventos en Cola Asíncrona", msg["events_published"], f"Backend: {msg['last_backend_used']}")

        st.json(metrics)
