# streamlit_app/app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import pydeck as pdk

load_dotenv()
API_URL = os.environ.get("BACKEND_API", "http://localhost:8000")

st.set_page_config(page_title="ARGO RAG Explorer", layout="wide")
st.title("ARGO RAG — Ask the ARGO_D DB (read-only)")

if "chat" not in st.session_state:
    st.session_state.chat = []

def post_question(q, top_k=5):
    payload = {"question": q, "top_k": top_k}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

with st.form("ask_form"):
    q = st.text_input("Ask a question about ARGO_D (read-only):")
    top_k = st.number_input("context (top-k summaries)", value=5, min_value=1, max_value=20)
    submitted = st.form_submit_button("Ask")
    if submitted and q:
        with st.spinner("Querying..."):
            try:
                resp = post_question(q, top_k=top_k)
            except Exception as e:
                st.error(f"Request failed: {e}")
                resp = None
        if resp:
            st.session_state.chat.append((q, resp))

# Render chat
for idx, (question, resp) in enumerate(reversed(st.session_state.chat[-10:])):
    st.markdown(f"**You:** {question}")
    if resp['type'] == 'sql':
        st.markdown("**Generated SQL (read-only)**")
        with st.expander("SQL"):
            st.code(resp['sql'], language='sql')
        st.markdown("**Retrieved context (top)**")
        for r in resp['retrieved_context']:
            st.write(f"- {r['source_table']} / {r['source_id']} (score ~ {r.get('score'):.3f}) — {r['summary_text'][:200]}")

        # show results
        if resp['results']:
            df = pd.DataFrame(resp['results'])
            st.dataframe(df)
            # simple geospatial plot detection
            cols = [c.lower() for c in df.columns]
            lat_col = None
            lon_col = None
            for c in df.columns:
                lc = c.lower()
                if 'lat' in lc and lat_col is None:
                    lat_col = c
                if 'lon' in lc or 'long' in lc:
                    lon_col = c
            if lat_col and lon_col:
                st.markdown("**Map**")
                df_map = df[[lon_col, lat_col]].dropna()
                df_map.columns = ['lon', 'lat']
                view = pdk.ViewState(latitude=df_map['lat'].mean(), longitude=df_map['lon'].mean(), zoom=4, pitch=0)
                layer = pdk.Layer('ScatterplotLayer', data=df_map.to_dict(orient='records'), get_position='[lon, lat]', get_radius=10000)
                r = pdk.Deck(layers=[layer], initial_view_state=view)
                st.pydeck_chart(r)

    else:
        st.markdown("**Answer**")
        st.write(resp['answer'])
        st.markdown("**Retrieved context (top)**")
        for r in resp['retrieved_context']:
            st.write(f"- {r['source_table']} / {r['source_id']} — {r['summary_text'][:200]}")
