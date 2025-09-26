"""
Dash Frontend for ARGO RAG + MCP
- Interactive chat UI
- Uses /query for RAG (Gemini SQL/text)
- Uses /mcp/* to manage a session and tools
- Context + SQL display, copy-to-clipboard, CSV download
- Rich visualizations: table, chart, map (auto-detect lat/lon)

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
# Config
# ----------------------
load_dotenv()
API_URL = os.environ.get("BACKEND_API", "http://localhost:8080")
THEME = os.environ.get("DASH_THEME", "SANDSTONE")  # try BOOTSTRAP, SLATE, CYBORG, etc.
THEME_MAP = {
    "SANDSTONE": dbc.themes.SANDSTONE,
    "BOOTSTRAP": dbc.themes.BOOTSTRAP,
    "SLATE": dbc.themes.SLATE,
    "CYBORG": dbc.themes.CYBORG,
}

external_stylesheets = [THEME_MAP.get(THEME, dbc.themes.SANDSTONE)]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="ARGO RAG Explorer")
server = app.server

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


def df_to_datatable(df: pd.DataFrame) -> dash_table.DataTable:
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "0.9rem"},
        style_header={"fontWeight": "bold"},
        id={"type": "result-table", "index": str(uuid.uuid4())},
    )


def build_map(df: pd.DataFrame):
    lat_col, lon_col = detect_lat_lon_columns(df.columns)
    if not (lat_col and lon_col):
        return None
    dmap = df[[lon_col, lat_col]].dropna()
    if dmap.empty:
        return None
    dmap = dmap.rename(columns={lon_col: "lon", lat_col: "lat"})
    fig = px.scatter_mapbox(dmap, lat="lat", lon="lon", zoom=3, height=420)
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    return fig


def bytes_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ----------------------
# Reusable UI blocks
# ----------------------
brand = html.Div([
    html.H2("ARGO RAG", className="mb-0"),
    html.Div("Ask the ARGO_D DB (read-only)", className="text-muted")
])

health_alert = dbc.Alert(id="health-alert", is_open=False, duration=4000)

def make_message(role: str, content: Component):
    color = "light" if role == "user" else ("primary" if role == "assistant" else "secondary")
    header = ("You" if role == "user" else ("Assistant" if role == "assistant" else "System"))
    return dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody(content)
    ], color=None, className="mb-3 shadow-sm")

controls = dbc.Card([
    dbc.CardHeader("Ask a question"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Question"),
                dcc.Textarea(id="question", placeholder="Ask a question about ARGO_D (read-only)", style={"width": "100%", "height": 80})
            ], md=8),
            dbc.Col([
                dbc.Label("Context (top-k)"),
                dbc.Input(id="topk", type="number", min=1, max=20, step=1, value=6),
                dbc.Label("Year (optional)", className="mt-2"),
                dbc.Input(id="year", type="number", min=2001, max=2017, step=1),
            ], md=4)
        ], className="g-2"),
        dbc.Row([
            dbc.Col([
                dbc.Button("Ask", id="ask-btn", color="primary", className="me-2"),
                dbc.Button("Test Backend", id="health-btn", color="secondary", outline=True, className="me-2"),
                dbc.Button("Describe Schema", id="schema-btn", color="info", outline=True, className="me-2"),
                dbc.Switch(id="theme-toggle", label="Dark mode", value=(THEME in ["SLATE", "CYBORG"]))
            ], width="auto")
        ], className="mt-2")
    ])
], className="shadow-sm rounded-3")

store = dcc.Store(id="chat-store", storage_type="memory", data={"items": [], "session_id": None})

chat_container = html.Div(id="chat-view", className="mt-3")

footer = html.Footer([
    html.Small(["Backend API: ", html.Code(API_URL)], className="text-muted")
], className="mt-4")

# ----------------------
# Layout
# ----------------------
app.layout = dbc.Container([
    dbc.Navbar([
        dbc.Container([dbc.NavbarBrand(brand, className="ms-0")])
    ], color="light", sticky="top", className="mb-3 shadow-sm"),

    health_alert,

    dbc.Row([dbc.Col(controls, md=12)]),

    dcc.Loading(chat_container, type="default"),

    store,
    footer
], fluid=True, className="pt-3 pb-4")

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
    Input("chat-store", "data"),
)
def render_chat(data):
    items = (data or {}).get("items", [])
    if not items:
        return html.Div("Ask something about ARGO_D to get started.", className="text-muted mt-3")

    cards = []
    for msg in items[-20:]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            cards.append(make_message("user", html.Div(content)))
        else:
            # assistant content can be schema text or query response
            if isinstance(content, dict) and content.get("type") == "schema":
                cards.append(make_message("assistant", html.Pre(content.get("text", ""), className="bg-light p-2 rounded")))
            elif isinstance(content, dict) and content.get("type") in ("sql", "text"):
                cards.append(render_answer_card(content))
            else:
                # fallback
                cards.append(make_message("assistant", html.Pre(json.dumps(content, indent=2))))
    return html.Div(cards)

# Render an answer card with tabs (SQL/Context/Results)

# add near other imports at the top:
# import base64

def render_answer_card(resp: dict):
    """
    Render a single assistant response as a card with tabs:
    - Context (retrieved chunks)
    - SQL (if type=sql) with Copy button
    - Table (results)
    - Chart (quick histogram)
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
                                    sql_text, id=sql_code_id, className="bg-light p-2 rounded"
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
        table = df_to_datatable(df) if not df.empty else html.Div("No rows", className="text-muted")
        tabs.append(
            dbc.Tab(
                dbc.Card(dbc.CardBody(table)),
                label="Table",
            )
        )

        # Chart tab (first numeric column, quick histogram)
        # Chart tab (choose the right plot)
        chart = html.Div("No numeric columns")
        if not df.empty:
            lower_cols = {c.lower(): c for c in df.columns}  # case-insensitive map

            # prefer depth profile if available
            if "depth_bin" in lower_cols and ("mean_salinity" in lower_cols or "avg_salinity" in lower_cols):
                depth_col = lower_cols["depth_bin"]
                sal_col = lower_cols.get("mean_salinity") or lower_cols["avg_salinity"]
                dfx = df[[depth_col, sal_col]].dropna().copy()
                # clamp to realistic depths (0–11000 m) to avoid outliers messing up the axis
                dfx = dfx[(dfx[depth_col] >= 0) & (dfx[depth_col] <= 11000)]
                dfx.sort_values(by=depth_col, inplace=True)
                fig = px.line(dfx, x=depth_col, y=sal_col, markers=True)
                chart = dcc.Graph(figure=fig)

            elif "depth" in lower_cols and ("mean_salinity" in lower_cols or "avg_salinity" in lower_cols or "salinity" in lower_cols):
                depth_col = lower_cols["depth"]
                sal_col = lower_cols.get("mean_salinity") or lower_cols.get("avg_salinity") or lower_cols["salinity"]
                dfx = df[[depth_col, sal_col]].dropna().copy()
                dfx = dfx[(dfx[depth_col] >= 0) & (dfx[depth_col] <= 11000)]
                dfx.sort_values(by=depth_col, inplace=True)
                fig = px.line(dfx, x=depth_col, y=sal_col, markers=True)
                chart = dcc.Graph(figure=fig)

            else:
                # fall back to a simple histogram of the first numeric column
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    fig = px.histogram(df, x=num_cols[0])
                    chart = dcc.Graph(figure=fig)

        tabs.append(dbc.Tab(dbc.Card(dbc.CardBody(chart)), label="Chart"))

        # Map tab (auto detect lat/lon)
        if not df.empty:
            fig_map = build_map(df)
            if fig_map is not None:
                tabs.append(
                    dbc.Tab(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_map))),
                        label="Map",
                    )
                )

        # Export tab (CSV as data URL)
        if not df.empty:
            csv_bytes = bytes_csv(df)  # you already have bytes_csv(df) defined
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
    return dbc.Card(
        [
            dbc.CardHeader("Assistant"),
            dbc.CardBody(dbc.Tabs(tabs)),
        ],
        className="mb-3 shadow-sm",
    )


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8501)), debug=True)
