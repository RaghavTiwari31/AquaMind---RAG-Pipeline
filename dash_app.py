import os
import json
import requests
from dotenv import load_dotenv

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx
from dash.dependencies import MATCH
import dash_bootstrap_components as dbc
from dash import dash_table
import plotly.express as px
import pandas as pd

# ----------------------
# Config
# ----------------------
load_dotenv()
API_URL = os.environ.get("BACKEND_API", "http://localhost:8080")

# ----------------------
# App Init (Dash + Bootstrap theme)
# ----------------------
external_stylesheets = [dbc.themes.SANDSTONE]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="ARGO RAG Explorer")
server = app.server  # for deployments that need `server`

# ----------------------
# Components
# ----------------------
brand = html.Div([
    html.H2("ARGO RAG", className="mb-0"),
    html.Div("Ask the ARGO_D DB (read-only)", className="text-muted")
])

health_alert = dbc.Alert(id="health-alert", is_open=False, duration=4000)

controls = dbc.Card([
    dbc.CardHeader("Ask a question"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Question"),
                dbc.Input(id="question", type="text", placeholder="Ask a question about ARGO_D (read-only)")
            ], md=8),
            dbc.Col([
                dbc.Label("Context (top-k summaries)"),
                dbc.Input(id="topk", type="number", min=1, max=20, step=1, value=5)
            ], md=4)
        ], className="g-2"),
        dbc.Row([
            dbc.Col([
                dbc.Button("Ask", id="ask-btn", color="primary", className="me-2"),
                dbc.Button("Test Backend", id="health-btn", color="secondary", outline=True)
            ], width="auto")
        ], className="mt-2")
    ])
], className="shadow-sm rounded-3")

# Store chat state in the browser
store = dcc.Store(id="chat-store", storage_type="memory", data={"items": []})

chat_container = html.Div(id="chat-view", className="mt-3")

footer = html.Footer([
    html.Small([
        "Backend API: ", html.Code(API_URL)
    ], className="text-muted")
], className="mt-4")

# ----------------------
# Layout
# ----------------------
app.layout = dbc.Container([
    dbc.Navbar([
        dbc.Container([
            dbc.NavbarBrand(brand, className="ms-0"),
        ])
    ], color="light", sticky="top", className="mb-3 shadow-sm"),

    health_alert,

    dbc.Row([
        dbc.Col(controls, md=12)
    ]),

    dcc.Loading(chat_container, type="default"),

    store,
    footer
], fluid=True, className="pt-3 pb-4")

# ----------------------
# Helpers
# ----------------------

def post_question(q: str, top_k: int = 5):
    payload = {"question": q, "top_k": top_k}
    try:
        r = requests.post(f"{API_URL}/query", json=payload, timeout=240)
        if r.status_code == 200:
            return r.json(), None
        else:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return None, f"Backend error ({r.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is it running on %s?" % API_URL
    except requests.exceptions.Timeout:
        return None, "Request timed out after 240 seconds"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def render_context_list(context_items):
    if not context_items:
        return html.Div("No retrieved context.", className="text-muted")

    accordion_items = []
    for i, r in enumerate(context_items, start=1):
        body = html.Div([
            html.Div([
                html.Strong("Source Table: "), html.Code(str(r.get("source_table"))),
                html.Span(" · "),
                html.Strong("ID: "), html.Code(str(r.get("source_id"))),
                html.Span(" · "),
                html.Strong("Score: "), html.Code(str(r.get("score", "N/A")))
            ], className="mb-2"),
            html.Pre((r.get("summary_text") or "")[:300] + ("..." if len(r.get("summary_text", "")) > 300 else ""),
                     className="bg-light p-2 rounded")
        ])
        accordion_items.append(
            dbc.AccordionItem(body, title=f"{i}. {r.get('source_table')} (ID: {r.get('source_id')})")
        )

    return dbc.Accordion(accordion_items, start_collapsed=True, flush=False, className="mt-2")


def detect_lat_lon_columns(columns):
    lat_col = None
    lon_col = None
    for c in columns:
        lc = c.lower()
        if ("lat" in lc) and (lat_col is None):
            lat_col = c
        if ("lon" in lc or "long" in lc) and (lon_col is None):
            lon_col = c
    return lat_col, lon_col


def render_results_table_and_map(results, columns):
    if not results:
        return html.Div([
            html.Div("No results returned from the query", className="text-info")
        ])

    df = pd.DataFrame(results, columns=columns or [])

    table = dash_table.DataTable(
        id={"type": "results-table", "index": 0},
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "0.9rem"},
        style_header={"fontWeight": "bold"}
    )

    lat_col, lon_col = detect_lat_lon_columns(list(df.columns))
    map_fig = None
    if lat_col and lon_col:
        df_map = df[[lon_col, lat_col]].dropna()
        if len(df_map) > 0:
            df_map = df_map.rename(columns={lon_col: "lon", lat_col: "lat"})
            map_fig = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                zoom=4,
                height=420
            )
            map_fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))

    cards = [
        dbc.Card([
            dbc.CardHeader("Query Results"),
            dbc.CardBody([table])
        ], className="mb-3 shadow-sm")
    ]

    if map_fig is not None:
        cards.append(
            dbc.Card([
                dbc.CardHeader("Map Visualization"),
                dbc.CardBody([dcc.Graph(figure=map_fig)])
            ], className="mb-3 shadow-sm")
        )

    return html.Div(cards)


def render_one_exchange(question, resp):
    header = html.Div([
        html.Strong("You:"), html.Span(" "), html.Span(question)
    ], className="mb-2")

    blocks = []

    if resp.get("type") == "sql":
        sql = resp.get("sql") or ""
        sql_card = dbc.Card([
            dbc.CardHeader("Generated SQL (read-only)"),
            dbc.CardBody([
                dbc.Collapse(html.Pre(sql, className="bg-light p-2 rounded"), is_open=False, id={"type": "sql-collapse", "index": id(sql)}),
                dbc.Button("Show/Hide SQL", id={"type": "sql-btn", "index": id(sql)}, size="sm", color="secondary", outline=True, className="mb-2"),
            ])
        ], className="mb-3 shadow-sm")

        context_card = dbc.Card([
            dbc.CardHeader("Retrieved Context"),
            dbc.CardBody([render_context_list(resp.get("retrieved_context", []))])
        ], className="mb-3 shadow-sm")

        results_block = render_results_table_and_map(resp.get("results", []), resp.get("columns", []))

        blocks.extend([sql_card, context_card, results_block])

    else:
        answer = resp.get("answer", "")
        answer_card = dbc.Card([
            dbc.CardHeader("Answer"),
            dbc.CardBody([html.Div(answer)])
        ], className="mb-3 shadow-sm")

        context_card = dbc.Card([
            dbc.CardHeader("Retrieved Context"),
            dbc.CardBody([render_context_list(resp.get("retrieved_context", []))])
        ], className="mb-3 shadow-sm")

        blocks.extend([answer_card, context_card])

    return dbc.Card([
        dbc.CardBody([header] + blocks)
    ], className="mb-3 rounded-3 shadow-sm")


# ----------------------
# Callbacks
# ----------------------

# Toggle SQL collapse
@app.callback(
    Output({"type": "sql-collapse", "index": MATCH}, "is_open"),
    Input({"type": "sql-btn", "index": MATCH}, "n_clicks"),
    State({"type": "sql-collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True
)
def _toggle_sql(n, is_open):
    if n:
        return not (is_open or False)
    return is_open


@app.callback(
    Output("health-alert", "children"),
    Output("health-alert", "color"),
    Output("health-alert", "is_open"),
    Input("health-btn", "n_clicks"),
    prevent_initial_call=True
)
def check_backend(_):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return ["✅ Backend is running and healthy!", "success", True]
        else:
            return [f"❌ Backend returned status {r.status_code}", "danger", True]
    except Exception as e:
        return [f"❌ Cannot connect to backend: {e}", "danger", True]


@app.callback(
    Output("chat-store", "data"),
    Output("health-alert", "children", allow_duplicate=True),
    Output("health-alert", "color", allow_duplicate=True),
    Output("health-alert", "is_open", allow_duplicate=True),
    Input("ask-btn", "n_clicks"),
    State("question", "value"),
    State("topk", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def on_ask(n_clicks, question, topk, data):
    data = data or {"items": []}
    if not question or str(question).strip() == "":
        return data, "Question cannot be empty", "warning", True

    resp, err = post_question(question, int(topk or 5))
    if err:
        return data, err, "danger", True

    items = data.get("items", [])
    items.append({"q": question, "resp": resp})
    return {"items": items[-20:]}, "", "info", False


@app.callback(
    Output("chat-view", "children"),
    Input("chat-store", "data")
)
def render_chat(data):
    items = (data or {}).get("items", [])
    if not items:
        return html.Div(
            "Ask something about ARGO_D to get started.",
            className="text-muted mt-3"
        )

    cards = []
    for item in items[-10:]:
        cards.append(render_one_exchange(item.get("q"), item.get("resp")))
    return html.Div(cards)


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8501)), debug=True)