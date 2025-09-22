# streamlit_app/app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import pydeck as pdk

load_dotenv()
API_URL = os.environ.get("BACKEND_API", "http://localhost:8080")

st.set_page_config(page_title="ARGO RAG Explorer", layout="wide")
st.title("ARGO RAG — Ask the ARGO_D DB (read-only)")

if "chat" not in st.session_state:
    st.session_state.chat = []

def post_question(q, top_k=5):
    payload = {"question": q, "top_k": top_k}
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Show the actual error from the backend
            try:
                error_detail = response.json()
                st.error(f"Backend error ({response.status_code}): {error_detail.get('detail', 'Unknown error')}")
            except:
                st.error(f"Backend error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is it running on http://localhost:8080?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out after 120 seconds")
        return None
    except Exception as e:
        st.error(f"🔥 Unexpected error: {e}")
        return None

# Test backend connection on startup
if st.button("🔍 Test Backend Connection"):
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend is running and healthy!")
            st.json(health_response.json())
        else:
            st.error(f"❌ Backend returned status {health_response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {e}")

with st.form("ask_form"):
    q = st.text_input("Ask a question about ARGO_D (read-only):")
    top_k = st.number_input("context (top-k summaries)", value=5, min_value=1, max_value=20)
    submitted = st.form_submit_button("Ask")
    
    if submitted and q:
        with st.spinner("Querying..."):
            resp = post_question(q, top_k=top_k)
            
        if resp:
            st.session_state.chat.append((q, resp))

# Render chat
for idx, (question, resp) in enumerate(reversed(st.session_state.chat[-10:])):
    st.markdown(f"**You:** {question}")
    
    if resp['type'] == 'sql':
        st.markdown("**Generated SQL (read-only)**")
        with st.expander("SQL Query"):
            st.code(resp['sql'], language='sql')
            
        st.markdown("**Retrieved Context**")
        with st.expander("Context Details"):
            for i, r in enumerate(resp['retrieved_context']):
                st.write(f"{i+1}. **{r['source_table']}** (ID: {r['source_id']}) - Score: {r.get('score', 'N/A')}")
                st.write(f"   {r['summary_text'][:300]}{'...' if len(r['summary_text']) > 300 else ''}")
                st.write("---")

        # Show results
        if resp['results']:
            st.markdown("**Query Results**")
            df = pd.DataFrame(resp['results'], columns=resp.get('columns', []))
            st.dataframe(df)
            
            # Simple geospatial plot detection
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
                st.markdown("**Map Visualization**")
                try:
                    df_map = df[[lon_col, lat_col]].dropna()
                    if len(df_map) > 0:
                        df_map.columns = ['lon', 'lat']
                        view = pdk.ViewState(
                            latitude=df_map['lat'].mean(), 
                            longitude=df_map['lon'].mean(), 
                            zoom=4, 
                            pitch=0
                        )
                        layer = pdk.Layer(
                            'ScatterplotLayer', 
                            data=df_map.to_dict(orient='records'), 
                            get_position='[lon, lat]', 
                            get_radius=10000,
                            get_color=[200, 30, 0, 160]
                        )
                        r = pdk.Deck(layers=[layer], initial_view_state=view)
                        st.pydeck_chart(r)
                    else:
                        st.info("No valid coordinates found for mapping")
                except Exception as e:
                    st.warning(f"Could not create map: {e}")
        else:
            st.info("No results returned from the query")

    else:
        st.markdown("**Answer**")
        st.write(resp['answer'])
        
        st.markdown("**Retrieved Context**")
        with st.expander("Context Details"):
            for i, r in enumerate(resp['retrieved_context']):
                st.write(f"{i+1}. **{r['source_table']}** (ID: {r['source_id']})")
                st.write(f"   {r['summary_text'][:300]}{'...' if len(r['summary_text']) > 300 else ''}")
                st.write("---")

    st.markdown("---")