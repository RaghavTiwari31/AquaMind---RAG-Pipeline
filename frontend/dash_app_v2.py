"""
FloatChat Frontend for ARGO RAG + MCP
- Interactive chat UI with dark mode
- Uses /query for RAG (Gemini SQL/text)
- Uses /mcp/* to manage a session and tools
- Context + SQL display, copy-to-clipboard, CSV download
- Rich visualizations: table, chart, map (auto-detect lat/lon)
- Custom design system with tiempos font and limited color palette

Env:
  BACKEND_API=http://localhost:8080
Run:
  python frontend/dash_app.py
"""
import os
import io
import json
import uuid
import requests
from dotenv import load_dotenv
import base64
import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
from dash.dependencies import MATCH
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import plotly.express as px
from dash.development.base_component import Component

# ----------------------
# Config & Styling
# ----------------------
load_dotenv()
API_URL = os.environ.get("BACKEND_API", "http://localhost:8080")

# Custom CSS with tiempos font and color palette - Modern UI
custom_css = """
/* Import tiempos font fallback */
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400;1,600&display=swap');

:root {
    --color-dark-grey: #1f1e1d;
    --color-light-grey: #262624;
    --color-orange: #d97757;
    --color-white: #ffffff;
    --color-black: #000000;
}

/* Base font family */
body, .dash-table-container, .dash-bootstrap {
    font-family: "tiempos", "tiempos Fallback", "Crimson Text", ui-serif, Georgia, Cambria, "Times New Roman", Times, serif !important;
    transition: all 0.3s ease;
}

/* Modern glassmorphism effects */
.glass-card {
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
}

.modern-shadow {
    box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.modern-border {
    border: none;
    border-radius: 12px;
}

/* Light mode styles */
.light-mode {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    color: var(--color-black);
    min-height: 100vh;
}

.light-mode .navbar {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    border: none;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.light-mode .card {
    background: var(--color-white);
    border: 1px solid rgba(38, 38, 36, 0.1);
    color: var(--color-black);
    border-radius: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.light-mode .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.light-mode .card-header {
    background: linear-gradient(135deg, var(--color-light-grey), var(--color-dark-grey));
    color: var(--color-white);
    border: none;
    border-radius: 12px 12px 0 0;
    padding: 1rem 1.5rem;
    font-weight: 600;
}

.light-mode .btn-primary {
    background: linear-gradient(135deg, var(--color-orange), #c86a4a);
    border: none;
    color: var(--color-white);
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(217, 119, 87, 0.4);
}

.light-mode .btn-primary:hover {
    background: linear-gradient(135deg, #c86a4a, var(--color-dark-grey));
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(217, 119, 87, 0.6);
}

.light-mode .btn-secondary, .light-mode .btn-info {
    background: var(--color-white);
    border: 2px solid var(--color-light-grey);
    color: var(--color-light-grey);
    border-radius: 8px;
    transition: all 0.2s ease;
}

.light-mode .btn-secondary:hover, .light-mode .btn-info:hover {
    background: var(--color-light-grey);
    color: var(--color-white);
    transform: translateY(-1px);
}

.light-mode .form-control, .light-mode .form-select, .light-mode textarea {
    border: 2px solid rgba(38, 38, 36, 0.1);
    background: var(--color-white);
    color: var(--color-black);
    border-radius: 8px;
    transition: all 0.2s ease;
    padding: 0.75rem;
}

.light-mode .form-control:focus, .light-mode .form-select:focus, .light-mode textarea:focus {
    border-color: var(--color-orange);
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.1);
    transform: scale(1.02);
}

/* Dark mode styles */
.dark-mode {
    background: linear-gradient(135deg, var(--color-dark-grey) 0%, #0a0a09 100%);
    color: var(--color-white);
    min-height: 100vh;
}

.dark-mode .navbar {
    background: rgba(31, 30, 29, 0.95) !important;
    backdrop-filter: blur(10px);
    border: none;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
}

.dark-mode .navbar-brand {
    color: var(--color-white) !important;
}

.dark-mode .card {
    background: rgba(38, 38, 36, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--color-white);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.dark-mode .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.dark-mode .card-header {
    background: linear-gradient(135deg, var(--color-dark-grey), #000);
    color: var(--color-white);
    border: none;
    border-radius: 12px 12px 0 0;
    padding: 1rem 1.5rem;
    font-weight: 600;
}

.dark-mode .btn-primary {
    background: linear-gradient(135deg, var(--color-orange), #c86a4a);
    border: none;
    color: var(--color-white);
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(217, 119, 87, 0.4);
}

.dark-mode .btn-primary:hover {
    background: linear-gradient(135deg, #c86a4a, var(--color-orange));
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(217, 119, 87, 0.6);
}

.dark-mode .btn-secondary, .dark-mode .btn-info {
    background: rgba(38, 38, 36, 0.8);
    border: 2px solid rgba(255, 255, 255, 0.2);
    color: var(--color-white);
    border-radius: 8px;
    transition: all 0.2s ease;
}

.dark-mode .btn-secondary:hover, .dark-mode .btn-info:hover {
    background: var(--color-orange);
    border-color: var(--color-orange);
    transform: translateY(-1px);
}

.dark-mode .form-control, .dark-mode .form-select, .dark-mode textarea {
    background: rgba(31, 30, 29, 0.8);
    border: 2px solid rgba(255, 255, 255, 0.1);
    color: var(--color-white);
    border-radius: 8px;
    transition: all 0.2s ease;
    padding: 0.75rem;
}

.dark-mode .form-control:focus, .dark-mode .form-select:focus, .dark-mode textarea:focus {
    background: rgba(31, 30, 29, 0.9);
    border-color: var(--color-orange);
    color: var(--color-white);
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.2);
    transform: scale(1.02);
}

.dark-mode .alert {
    background: rgba(38, 38, 36, 0.9);
    border: 1px solid var(--color-orange);
    color: var(--color-white);
    border-radius: 8px;
    backdrop-filter: blur(10px);
}

.dark-mode pre {
    background: rgba(31, 30, 29, 0.9) !important;
    color: var(--color-white);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

.dark-mode .text-muted {
    color: rgba(255, 255, 255, 0.6) !important;
}

/* Dash table styling */
.dash-table-container {
    border-radius: 8px;
    overflow: hidden;
}

.dark-mode .dash-table-container {
    background: rgba(38, 38, 36, 0.8);
}

.dark-mode .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
    background: rgba(38, 38, 36, 0.8);
    color: var(--color-white);
}

.dark-mode .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
    background: var(--color-dark-grey) !important;
    color: var(--color-white) !important;
}

.dark-mode .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
    background: rgba(38, 38, 36, 0.6) !important;
    color: var(--color-white) !important;
}

/* Modern switch styling */
.form-check-input {
    width: 3rem;
    height: 1.5rem;
    border-radius: 2rem;
    background-color: rgba(38, 38, 36, 0.3);
    border: 2px solid rgba(38, 38, 36, 0.2);
    transition: all 0.3s ease;
}

.form-check-input:checked {
    background-color: var(--color-orange);
    border-color: var(--color-orange);
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.2);
}

.form-check-input:focus {
    border-color: var(--color-orange);
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.1);
}

/* Modern tab styling */
.nav-tabs .nav-link {
    border: none;
    border-radius: 8px 8px 0 0;
    margin-right: 0.25rem;
    transition: all 0.2s ease;
}

.nav-tabs .nav-link.active {
    background: var(--color-orange);
    color: var(--color-white);
}

.nav-tabs .nav-link:hover {
    background: rgba(217, 119, 87, 0.1);
}

/* Footer styling */
.dark-mode footer {
    color: rgba(255, 255, 255, 0.5);
}

.light-mode footer {
    color: rgba(0, 0, 0, 0.6);
}

/* Loading animation */
._dash-loading {
    background: var(--color-orange);
    border-radius: 4px;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1);
}

::-webkit-scrollbar-thumb {
    background: var(--color-orange);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #c86a4a;
}
"""

external_stylesheets = [dbc.themes.BOOTSTRAP]
app: Dash = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets, 
    title="FloatChat - ARGO Explorer"
)
server = app.server

# Add custom CSS
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {custom_css}
        </style>
    </head>
    <body class="light-mode" id="app-body">
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# ----------------------
# Helpers
# ----------------------

def api_get(path: str, timeout=30):
    return requests.get(f"{API_URL}{path}", timeout=timeout)


def api_post(path: str, payload: dict, timeout=240):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)


def detect_lat_lon_columns(columns):
    lat_col = None
    lon_col = None
    for c in columns:
        lc = c.lower()
        if lat_col is None and "lat" in lc:
            lat_col = c
        if lon_col is None and ("lon" in lc or "long" in lc):
            lon_col = c
    return lat_col, lon_col


def df_to_datatable(df: pd.DataFrame, is_dark_mode: bool = False) -> dash_table.DataTable:
    style_data = {
        'backgroundColor': '#262624' if is_dark_mode else 'white',
        'color': 'white' if is_dark_mode else 'black',
    }
    style_header = {
        'backgroundColor': '#1f1e1d' if is_dark_mode else '#262624',
        'color': 'white',
        'fontWeight': 'bold'
    }
    
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontFamily": '"tiempos", "tiempos Fallback", "Crimson Text", ui-serif, Georgia, Cambria, "Times New Roman", Times, serif', "fontSize": "0.9rem"},
        style_data=style_data,
        style_header=style_header,
        id={"type": "result-table", "index": str(uuid.uuid4())},
    )


def build_map(df: pd.DataFrame, is_dark_mode: bool = False):
    lat_col, lon_col = detect_lat_lon_columns(df.columns)
    if not (lat_col and lon_col):
        return None
    dmap = df[[lon_col, lat_col]].dropna()
    if dmap.empty:
        return None
    dmap = dmap.rename(columns={lon_col: "lon", lat_col: "lat"})
    
    # Color the map markers with orange
    fig = px.scatter_mapbox(
        dmap, 
        lat="lat", 
        lon="lon", 
        zoom=3, 
        height=420,
        color_discrete_sequence=["#d97757"]
    )
    
    # Set map style based on theme
    map_style = "carto-darkmatter" if is_dark_mode else "open-street-map"
    bg_color = "#1f1e1d" if is_dark_mode else "white"
    
    fig.update_layout(
        mapbox_style=map_style, 
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color
    )
    return fig


def bytes_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ----------------------
# Reusable UI blocks
# ----------------------
brand = html.Div([
    html.H2("FloatChat", className="mb-0", style={"color": "#d97757"}),
    html.Div("Explore ARGO oceanographic data", className="text-muted", style={"textAlign": "left"})
], style={"marginLeft": "20px", "paddingLeft": "0px"})

health_alert = dbc.Alert(id="health-alert", is_open=False, duration=4000)

def make_message(role: str, content: Component, is_dark_mode: bool = False):
    if role == "user":
        # User messages - orange theme
        header_bg = "#d97757"
        body_bg = "rgba(217, 119, 87, 0.05)" if not is_dark_mode else "rgba(217, 119, 87, 0.15)"
        header_text = "#ffffff"
        body_text = "#000000" if not is_dark_mode else "#ffffff"
        header = "You"
    elif role == "assistant":
        # Assistant messages - grey theme
        header_bg = "#262624"
        body_bg = "#ffffff" if not is_dark_mode else "rgba(38, 38, 36, 0.3)"
        header_text = "#ffffff"
        body_text = "#000000" if not is_dark_mode else "#ffffff"
        header = "FloatChat"
    else:
        # System messages - dark theme
        header_bg = "#1f1e1d"
        body_bg = "#ffffff" if not is_dark_mode else "rgba(31, 30, 29, 0.3)"
        header_text = "#ffffff"
        body_text = "#000000" if not is_dark_mode else "#ffffff"
        header = "System"
    
    return dbc.Card([
        dbc.CardHeader(
            header, 
            className="modern-border", 
            style={
                "backgroundColor": header_bg, 
                "color": header_text, 
                "border": "none",
                "fontWeight": "600",
                "fontSize": "0.9rem",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }
        ),
        dbc.CardBody(
            content, 
            className="modern-border",
            style={
                "backgroundColor": body_bg, 
                "color": body_text,
                "border": "none"
            }
        )
    ], className="mb-3 modern-shadow modern-border")

controls = dbc.Card([
    dbc.CardHeader("Ask about ocean data", className="modern-border"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Question", style={"fontWeight": "500", "marginBottom": "0.5rem"}),
                dcc.Textarea(
                    id="question", 
                    placeholder="Ask about ARGO oceanographic data...", 
                    className="modern-border",
                    style={"width": "100%", "height": 80, "resize": "vertical"}
                )
            ], md=8),
            dbc.Col([
                dbc.Label("Context (top-k)", style={"fontWeight": "500", "marginBottom": "0.5rem"}),
                dbc.Input(id="topk", type="number", min=1, max=20, step=1, value=6, className="modern-border"),
                dbc.Label("Year (optional)", className="mt-3", style={"fontWeight": "500", "marginBottom": "0.5rem"}),
                dbc.Input(id="year", type="number", min=2001, max=2017, step=1, className="modern-border"),
            ], md=4)
        ], className="g-3"),
        html.Hr(style={"margin": "1.5rem 0", "border": "none", "height": "1px", "background": "linear-gradient(to right, transparent, rgba(217, 119, 87, 0.3), transparent)"}),
        dbc.Row([
            dbc.Col([
                dbc.Button("Ask", id="ask-btn", color="primary", className="me-3 modern-border", style={"minWidth": "100px"}),
                dbc.Button("Test Backend", id="health-btn", color="secondary", outline=True, className="me-3 modern-border"),
                dbc.Button("Describe Schema", id="schema-btn", color="info", outline=True, className="me-3 modern-border"),
            ], width="auto"),
            dbc.Col([
                html.Div([
                    dbc.Switch(
                        id="theme-toggle", 
                        label="Dark mode", 
                        value=False, 
                        className="modern-switch",
                        style={"transform": "scale(1.2)"}
                    )
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"})
            ], width=True)
        ], className="mt-3", align="center")
    ], className="modern-border")
], className="modern-shadow modern-border")

store = dcc.Store(id="chat-store", storage_type="memory", data={"items": [], "session_id": None})
theme_store = dcc.Store(id="theme-store", storage_type="session", data={"dark_mode": False})

chat_container = html.Div(id="chat-view", className="mt-3")

footer = html.Footer([
    html.Small(["Backend API: ", html.Code(API_URL)], className="text-muted")
], className="mt-4")

# ----------------------
# Layout
# ----------------------
app.layout = html.Div([
    dbc.Navbar(
    dbc.Container(
        dbc.NavbarBrand(brand, className="ms-0"),
        fluid=True,
        className="px-0 mx-0"   # remove left/right padding
    ),
    color="light",
    sticky="top",
    className="mb-3 shadow-sm px-0"
    ),

    dbc.Container([
        health_alert,
        dbc.Row([dbc.Col(controls, md=12)]),
        dcc.Loading(chat_container, type="default"),
        footer
    ], fluid=True, className="pt-3 pb-4"),

    store,
    theme_store
], id="main-container")

# ----------------------
# Theme Toggle Callback
# ----------------------
@app.callback(
    [Output("theme-store", "data"),
     Output("main-container", "className")],
    [Input("theme-toggle", "value")],
    [State("theme-store", "data")]
)
def toggle_theme(dark_mode, theme_data):
    theme_data = theme_data or {"dark_mode": False}
    theme_data["dark_mode"] = dark_mode
    
    container_class = "dark-mode" if dark_mode else "light-mode"
    
    return theme_data, container_class

# Add clientside callback to update body class
app.clientside_callback(
    """
    function(theme_data) {
        const body = document.getElementById('app-body');
        if (body && theme_data) {
            if (theme_data.dark_mode) {
                body.className = 'dark-mode';
            } else {
                body.className = 'light-mode';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-toggle", "persistence"),  # dummy output
    Input("theme-store", "data")
)

# ----------------------
# Callbacks: bootstrap & MCP session
# ----------------------

@app.callback(
    Output("chat-store", "data"),
    Input("chat-store", "modified_timestamp"),
    State("chat-store", "data"),
    prevent_initial_call=False,
)
def bootstrap_session(_, data):
    data = data or {"items": [], "session_id": None}
    if data.get("session_id"):
        return data
    # open MCP session
    try:
        r = api_post("/mcp/session.open", {})
        sid = r.json()["session_id"]
        data["session_id"] = sid
    except Exception:
        # allow UI to still function without MCP
        data["session_id"] = str(uuid.uuid4())
    return data

# Health check
@app.callback(
    Output("health-alert", "children"),
    Output("health-alert", "color"),
    Output("health-alert", "is_open"),
    Input("health-btn", "n_clicks"),
    prevent_initial_call=True,
)
def check_backend(_):
    try:
        r = api_get("/health", timeout=5)
        if r.status_code == 200:
            return ["✅ Backend is running and healthy!", "success", True]
        else:
            return [f"❌ Backend returned status {r.status_code}", "danger", True]
    except Exception as e:
        return [f"❌ Cannot connect to backend: {e}", "danger", True]

# Ask
@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("health-alert", "children", allow_duplicate=True),
    Output("health-alert", "color", allow_duplicate=True),
    Output("health-alert", "is_open", allow_duplicate=True),
    Input("ask-btn", "n_clicks"),
    State("question", "value"),
    State("topk", "value"),
    State("year", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def on_ask(_, question, topk, year, data):
    data = data or {"items": [], "session_id": None}
    if not question or not str(question).strip():
        return data, "Question cannot be empty", "warning", True

    payload = {"question": question, "top_k": int(topk or 6)}
    if year:
        payload["year"] = int(year)
    # include session id if backend wants to persist
    if data.get("session_id"):
        payload["session_id"] = data["session_id"]

    try:
        r = api_post("/query", payload)
        if r.status_code != 200:
            try:
                errmsg = r.json().get("detail", r.text)
            except Exception:
                errmsg = r.text
            return data, f"Backend error ({r.status_code}): {errmsg}", "danger", True
        resp = r.json()
    except Exception as e:
        return data, f"Request failed: {e}", "danger", True

    # append chat
    items = data.get("items", [])
    items.append({"role": "user", "content": question})
    items.append({"role": "assistant", "content": resp})

    # also append to MCP session log (best effort)
    sid = data.get("session_id")
    try:
        if sid:
            api_post("/mcp/session.append", {"session_id": sid, "role": "user", "content": question})
            api_post("/mcp/session.append", {"session_id": sid, "role": "assistant", "content": json.dumps(resp)})
    except Exception:
        pass

    data["items"] = items[-50:]
    return data, "", "info", False

# Describe schema via MCP
@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Input("schema-btn", "n_clicks"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def on_schema(_, data):
    data = data or {"items": [], "session_id": None}
    sid = data.get("session_id")
    try:
        r = api_post("/mcp/tools.call", {"session_id": sid, "tool": "db.describe_schema", "args": {}})
        result = r.json().get("result", {})
        schema_text = (result or {}).get("schema", "")
    except Exception as e:
        schema_text = f"(error) {e}"

    items = data.get("items", [])
    items.append({"role": "assistant", "content": {"type": "schema", "text": schema_text}})
    data["items"] = items[-50:]
    return data

# Render chat
@app.callback(
    Output("chat-view", "children"),
    [Input("chat-store", "data"),
     Input("theme-store", "data")],
)
def render_chat(data, theme_data):
    items = (data or {}).get("items", [])
    is_dark_mode = (theme_data or {}).get("dark_mode", False)
    
    if not items:
        return html.Div("Ask something about ARGO oceanographic data to get started.", className="text-muted mt-3")

    cards = []
    for msg in items[-20:]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            cards.append(make_message("user", html.Div(content), is_dark_mode))
        else:
            # assistant content can be schema text or query response
            if isinstance(content, dict) and content.get("type") == "schema":
                cards.append(make_message("assistant", html.Pre(content.get("text", ""), className="p-2 rounded"), is_dark_mode))
            elif isinstance(content, dict) and content.get("type") in ("sql", "text"):
                cards.append(render_answer_card(content, is_dark_mode))
            else:
                # fallback
                cards.append(make_message("assistant", html.Pre(json.dumps(content, indent=2)), is_dark_mode))
    return html.Div(cards)

def render_answer_card(resp: dict, is_dark_mode: bool = False):
    """
    Render a single assistant response as a card with tabs:
    - Context (retrieved chunks)
    - SQL (if type=sql) with Copy button
    - Table (results)
    - Chart (quick histogram/depth profile)
    - Map (auto lat/lon)
    - Export (CSV download)
    - or Answer (if type=text)
    """
    tabs = []

    # ---------- Context tab ----------
    ctx_list = resp.get("retrieved_context", []) or []
    ctx_children = []
    for i, r in enumerate(ctx_list, 1):
        snippet = (r.get("summary_text") or "")
        if len(snippet) > 400:
            snippet = snippet[:397] + "..."
        ctx_children.append(
            html.Li(
                [
                    html.Strong(f"{i}. {r.get('source_table')}"),
                    html.Span(" · "),
                    html.Code(str(r.get("source_id"))),
                    html.Div(snippet, className="text-muted"),
                ],
                className="mb-2",
            )
        )

    tabs.append(
        dbc.Tab(
            dbc.Card(
                dbc.CardBody(
                    html.Ul(ctx_children) if ctx_children else html.Div("None")
                )
            ),
            label="Context",
        )
    )

    # ---------- SQL / Results path ----------
    if resp.get("type") == "sql":
        sql_text = resp.get("sql") or ""
        sql_code_id = f"sql-code-{uuid.uuid4()}"

        # SQL tab (with copy)
        tabs.append(
            dbc.Tab(
                dbc.Card(
                    dbc.CardBody(
                        html.Div(
                            [
                                dbc.Button(
                                    "Copy SQL",
                                    id={"type": "copy-sql", "index": str(uuid.uuid4())},
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                    className="mb-2",
                                ),
                                dcc.Clipboard(target_id=sql_code_id, title="Copy"),
                                html.Pre(
                                    sql_text, id=sql_code_id, className="p-2 rounded",
                                    style={"backgroundColor": "#1f1e1d" if is_dark_mode else "#f8f9fa", "color": "white" if is_dark_mode else "black"}
                                ),
                            ]
                        )
                    )
                ),
                label="SQL",
            )
        )

        # Results dataframe
        df = pd.DataFrame(resp.get("results", []), columns=resp.get("columns", []))

        # Table tab
        table = df_to_datatable(df, is_dark_mode) if not df.empty else html.Div("No rows", className="text-muted")
        tabs.append(
            dbc.Tab(
                dbc.Card(dbc.CardBody(table)),
                label="Table",
            )
        )

        # Chart tab (depth profile or histogram)
        chart = html.Div("No numeric columns")
        if not df.empty:
            lower_cols = {c.lower(): c for c in df.columns}

            # Chart styling based on theme
            chart_bg = "#1f1e1d" if is_dark_mode else "white"
            chart_text = "white" if is_dark_mode else "black"
            
            # prefer depth profile if available
            if "depth_bin" in lower_cols and ("mean_salinity" in lower_cols or "avg_salinity" in lower_cols):
                depth_col = lower_cols["depth_bin"]
                sal_col = lower_cols.get("mean_salinity") or lower_cols["avg_salinity"]
                dfx = df[[depth_col, sal_col]].dropna().copy()
                dfx = dfx[(dfx[depth_col] >= 0) & (dfx[depth_col] <= 11000)]
                dfx.sort_values(by=depth_col, inplace=True)
                fig = px.line(dfx, x=depth_col, y=sal_col, markers=True, color_discrete_sequence=["#d97757"])
                fig.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font_color=chart_text)
                chart = dcc.Graph(figure=fig)

            elif "depth" in lower_cols and ("mean_salinity" in lower_cols or "avg_salinity" in lower_cols or "salinity" in lower_cols):
                depth_col = lower_cols["depth"]
                sal_col = lower_cols.get("mean_salinity") or lower_cols.get("avg_salinity") or lower_cols["salinity"]
                dfx = df[[depth_col, sal_col]].dropna().copy()
                dfx = dfx[(dfx[depth_col] >= 0) & (dfx[depth_col] <= 11000)]
                dfx.sort_values(by=depth_col, inplace=True)
                fig = px.line(dfx, x=depth_col, y=sal_col, markers=True, color_discrete_sequence=["#d97757"])
                fig.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font_color=chart_text)
                chart = dcc.Graph(figure=fig)

            else:
                # fall back to a simple histogram of the first numeric column
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    fig = px.histogram(df, x=num_cols[0], color_discrete_sequence=["#d97757"])
                    fig.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font_color=chart_text)
                    chart = dcc.Graph(figure=fig)

        tabs.append(dbc.Tab(dbc.Card(dbc.CardBody(chart)), label="Chart"))

        # Map tab (auto detect lat/lon)
        if not df.empty:
            fig_map = build_map(df, is_dark_mode)
            if fig_map is not None:
                tabs.append(
                    dbc.Tab(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_map))),
                        label="Map",
                    )
                )

        # Export tab (CSV as data URL)
        if not df.empty:
            csv_bytes = bytes_csv(df)
            b64 = base64.b64encode(csv_bytes).decode("utf-8")
            href = f"data:text/csv;base64,{b64}"
            tabs.append(
                dbc.Tab(
                    dbc.Card(
                        dbc.CardBody(
                            html.A(
                                "Download CSV",
                                href=href,
                                download="results.csv",
                                target="_blank",
                                className="btn",
                                style={"backgroundColor": "#d97757", "color": "white", "textDecoration": "none", "padding": "8px 16px", "borderRadius": "4px"}
                            )
                        )
                    ),
                    label="Export",
                )
            )

    else:
        # ---------- Text answer path ----------
        tabs.append(
            dbc.Tab(
                dbc.Card(dbc.CardBody(html.Div(resp.get("answer", "")))),
                label="Answer",
            )
        )

    # Wrap in a card with tabs
    card_bg = "#262624" if is_dark_mode else "white"
    header_bg = "#1f1e1d" if is_dark_mode else "#262624"
    text_color = "white" if is_dark_mode else "black"
    
    return dbc.Card(
        [
            dbc.CardHeader("FloatChat", style={"backgroundColor": header_bg, "color": "white", "border": "none"}),
            dbc.CardBody(dbc.Tabs(tabs), style={"backgroundColor": card_bg, "color": text_color}),
        ],
        className="mb-3 shadow-sm",
        style={"border": "none", "backgroundColor": card_bg}
    )


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    app.run(host="localhost", port=int(os.environ.get("PORT", 8501)), debug=True)