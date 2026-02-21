import json
import os
from pathlib import Path

import pandas as pd
from unidecode import unidecode

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "AfiliadosAtivos_EstadoCidade.xlsx"
GEO_MANIFEST_PATH = BASE_DIR / "data" / "geo" / "manifest.json"

VALUE_COL = "qt_influencers"
COL_ESTADO = "estado"
COL_CIDADE = "cidade"
MAP_STYLE = "carto-positron"


# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    return unidecode(str(s)).strip().lower()


def load_data():
    df = pd.read_excel(EXCEL_PATH)
    df["estado_norm"] = df[COL_ESTADO].map(norm)
    df["cidade_norm"] = df[COL_CIDADE].map(norm)
    return df


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def geojson_properties_df(geojson: dict):
    rows = []
    for feature in geojson.get("features", []):
        props = dict(feature.get("properties") or {})
        props["geo_id"] = str(feature.get("id", props.get("geo_id", "")))
        rows.append(props)
    return pd.DataFrame(rows)


def _extend_bounds(coords, current):
    if not coords:
        return current
    first = coords[0]
    if isinstance(first, (int, float)):
        lon, lat = coords
        minx, miny, maxx, maxy = current
        return min(minx, lon), min(miny, lat), max(maxx, lon), max(maxy, lat)
    for item in coords:
        current = _extend_bounds(item, current)
    return current


def geojson_bounds(geojson: dict):
    bounds = (999.0, 999.0, -999.0, -999.0)
    for feature in geojson.get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        bounds = _extend_bounds(coords, bounds)
    return bounds


def fit_zoom(bounds):
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny)
    if span > 25:
        return 3.5
    if span > 15:
        return 4.5
    if span > 8:
        return 5.5
    if span > 4:
        return 6.5
    if span > 2:
        return 7.5
    if span > 1:
        return 8.3
    return 9.2


def error_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color="#334155"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="white")
    return fig


# -----------------------------
# Runtime state (lazy + cache)
# -----------------------------
df = None
manifest = None
states_geojson = None
states_df = None
state_centroids = None

state_totals = None
city_agg_by_state = None
state_meta = None

muni_geojson_by_uf = None
muni_df_by_uf = None
pts_by_uf = None
bounds_by_uf = None

brazil_fig_cached = None
state_fig_cache = None
init_error = None


def load_uf_assets(uf: str):
    if uf in muni_geojson_by_uf:
        return

    muni_geo_path = BASE_DIR / manifest["municipalities_dir"] / f"{uf}.geojson"
    centroids_path = BASE_DIR / manifest["centroids_dir"] / f"{uf}_centroids.csv"

    muni_geo = read_json(muni_geo_path)
    muni_geojson_by_uf[uf] = muni_geo
    muni_df_by_uf[uf] = geojson_properties_df(muni_geo)
    pts_by_uf[uf] = pd.read_csv(centroids_path)
    bounds_by_uf[uf] = geojson_bounds(muni_geo)


def ensure_data_loaded():
    global df, manifest, states_geojson, states_df, state_centroids
    global state_totals, city_agg_by_state, state_meta
    global muni_geojson_by_uf, muni_df_by_uf, pts_by_uf, bounds_by_uf
    global brazil_fig_cached, state_fig_cache, init_error

    if df is not None or init_error is not None:
        return

    try:
        if not GEO_MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "Assets geográficos não encontrados. Rode: .venv/bin/python scripts/build_geo_assets.py"
            )

        manifest = read_json(GEO_MANIFEST_PATH)
        states_geo_path = BASE_DIR / manifest["states_geojson"]
        state_centroids_path = BASE_DIR / manifest["state_centroids_csv"]

        states_geojson = read_json(states_geo_path)
        states_df = geojson_properties_df(states_geojson)
        state_centroids = pd.read_csv(state_centroids_path)

        df = load_data()
        state_totals = df.groupby("estado_norm")[VALUE_COL].sum()

        city_agg = df.groupby(["estado_norm", "cidade_norm"], as_index=False)[VALUE_COL].sum()
        city_agg_by_state = {}
        for estado_norm, grp in city_agg.groupby("estado_norm"):
            city_agg_by_state[estado_norm] = grp.rename(
                columns={"cidade_norm": "muni_name_norm"}
            )[["muni_name_norm", VALUE_COL]]

        state_meta = {
            key: value for key, value in manifest.get("state_meta", {}).items()
        }

        muni_geojson_by_uf = {}
        muni_df_by_uf = {}
        pts_by_uf = {}
        bounds_by_uf = {}

        brazil_fig_cached = None
        state_fig_cache = {}

    except Exception as exc:
        init_error = str(exc)


def build_brazil_fig():
    agg_state = (
        df.groupby("estado_norm", as_index=False)[VALUE_COL]
        .sum()
        .rename(columns={"estado_norm": "state_name_norm"})
    )

    states_plot = states_df.merge(agg_state, on="state_name_norm", how="left")
    states_plot[VALUE_COL] = states_plot[VALUE_COL].fillna(0)

    fig = px.choropleth_mapbox(
        states_plot,
        geojson=states_geojson,
        locations="geo_id",
        featureidkey="id",
        color=VALUE_COL,
        custom_data=["state_name_norm"],
        color_continuous_scale="Sunsetdark",
        hover_name="name_state",
        hover_data={VALUE_COL: ":,.0f"},
        labels={VALUE_COL: "Quantidade"},
        opacity=0.88,
        mapbox_style=MAP_STYLE,
        center={"lat": -14.2, "lon": -51.9},
        zoom=3.4,
    )

    fig.update_traces(
        marker_line_width=0.8,
        marker_line_color="#F8FAFC",
        hovertemplate="<b>%{hovertext}</b><br>Quantidade: %{z:,.0f}<extra></extra>",
    )

    fig.update_layout(
        mapbox=dict(pitch=42, bearing=-14),
        coloraxis_colorbar=dict(title="Quantidade", thickness=12, len=0.78),
    )

    fig.update_layout(
        title="Brasil - Influencers por Estado (clique para detalhar)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
    )

    return fig


def build_state_fig(estado_norm: str):
    info = state_meta.get(estado_norm)
    if not info:
        return error_figure("Estado não encontrado.")

    uf = info["uf"]
    state_name = info["state_name"]
    load_uf_assets(uf)

    agg_city = city_agg_by_state.get(estado_norm)
    if agg_city is None:
        agg_city = pd.DataFrame(columns=["muni_name_norm", VALUE_COL])

    muni_state = muni_df_by_uf[uf].copy()
    muni_state = muni_state.merge(agg_city, on="muni_name_norm", how="left")
    muni_state[VALUE_COL] = muni_state[VALUE_COL].fillna(0)

    pts = pts_by_uf[uf].copy()
    pts = pts.merge(agg_city, on="muni_name_norm", how="left")
    pts[VALUE_COL] = pts[VALUE_COL].fillna(0)
    pts_plot = pts[pts[VALUE_COL] > 0].copy()

    fig = px.choropleth_mapbox(
        muni_state,
        geojson=muni_geojson_by_uf[uf],
        locations="geo_id",
        featureidkey="id",
        color=VALUE_COL,
        color_continuous_scale="Tealgrn",
        opacity=0.65,
        hover_name="name_muni",
        hover_data={VALUE_COL: ":,.0f"},
        mapbox_style=MAP_STYLE,
    )

    fig.update_traces(marker_line_width=0.4, marker_line_color="#F7F9FC")

    fig.add_trace(
        go.Scattermapbox(
            lat=pts_plot["lat"] - 0.06,
            lon=pts_plot["lon"] + 0.05,
            mode="markers",
            marker=dict(
                size=(pts_plot[VALUE_COL] ** 0.5) * 2.9 + 4,
                color="rgba(0, 0, 0, 0.18)",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=pts_plot["lat"],
            lon=pts_plot["lon"],
            text=pts_plot["name_muni"]
            + "<br>"
            + pts_plot[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", ".")),
            hovertemplate="%{text}<extra></extra>",
            mode="markers",
            marker=dict(
                size=(pts_plot[VALUE_COL] ** 0.5) * 2.4 + 4,
                color="#0F766E",
                opacity=0.86,
                line=dict(width=1.2, color="#ECFEFF"),
            ),
            showlegend=False,
        )
    )

    bounds = bounds_by_uf[uf]
    center = {"lat": (bounds[1] + bounds[3]) / 2, "lon": (bounds[0] + bounds[2]) / 2}
    zoom = fit_zoom(bounds)
    fig.update_layout(
        mapbox=dict(center=center, zoom=zoom, pitch=50, bearing=18),
        coloraxis_colorbar=dict(title="Qtd", thickness=10, len=0.7),
    )

    total_state = int(state_totals.get(estado_norm, 0))
    fig.update_layout(
        title=f"{state_name} - Cidades (total no estado: {total_state:,})".replace(",", "."),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        showlegend=False,
    )
    return fig


def get_brazil_fig():
    global brazil_fig_cached
    if brazil_fig_cached is None:
        brazil_fig_cached = build_brazil_fig()
    return brazil_fig_cached


def get_state_fig(estado_norm: str):
    if estado_norm in state_fig_cache:
        return state_fig_cache[estado_norm]
    fig = build_state_fig(estado_norm)
    state_fig_cache[estado_norm] = fig
    return fig


# -----------------------------
# App
# -----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "maxWidth": "1240px",
        "margin": "0 auto",
        "fontFamily": "Segoe UI, sans-serif",
        "padding": "20px 14px 26px 14px",
        "background": "linear-gradient(140deg, #f8fafc 0%, #e2e8f0 45%, #dbeafe 100%)",
        "borderRadius": "18px",
        "boxShadow": "0 10px 34px rgba(15, 23, 42, 0.12)",
    },
    children=[
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center", "paddingBottom": "10px"},
            children=[
                html.Button(
                    "Voltar para Brasil",
                    id="btn-back",
                    n_clicks=0,
                    style={
                        "border": "none",
                        "background": "#0f172a",
                        "color": "white",
                        "padding": "10px 14px",
                        "borderRadius": "10px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                    },
                ),
                html.Div(id="subtitle", style={"color": "#334155", "fontWeight": "600"}),
            ],
        ),
        dcc.Store(id="store-view", data={"level": "br", "estado_norm": None}),
        dcc.Graph(
            id="map",
            style={"height": "80vh", "borderRadius": "14px", "overflow": "hidden"},
            config={"displaylogo": False},
        ),
    ],
)


@app.callback(
    Output("map", "figure"),
    Output("store-view", "data"),
    Output("subtitle", "children"),
    Input("map", "clickData"),
    Input("btn-back", "n_clicks"),
    State("store-view", "data"),
)
def update_map(clickData, n_back, view):
    ensure_data_loaded()

    if init_error:
        msg = f"Erro ao carregar dados geográficos: {init_error}"
        return error_figure(msg), {"level": "br", "estado_norm": None}, "Erro de inicialização"

    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "btn-back":
        return get_brazil_fig(), {"level": "br", "estado_norm": None}, "Visão Brasil"

    if view["level"] == "br" and clickData:
        try:
            point = clickData["points"][0]
            estado_norm = None

            if point.get("customdata"):
                raw = point["customdata"]
                if isinstance(raw, (list, tuple)):
                    estado_norm = raw[0]
                else:
                    estado_norm = raw
            elif point.get("location") is not None:
                location = str(point["location"])
                match = states_df[states_df["geo_id"] == location]
                if not match.empty:
                    estado_norm = match.iloc[0]["state_name_norm"]

            if estado_norm:
                fig = get_state_fig(estado_norm)
                return fig, {"level": "state", "estado_norm": estado_norm}, "Visão Estado (clique em Voltar para Brasil)"
        except Exception:
            pass

    if view["level"] == "state" and view["estado_norm"]:
        fig = get_state_fig(view["estado_norm"])
        return fig, view, "Visão Estado (clique em Voltar para Brasil)"

    return get_brazil_fig(), {"level": "br", "estado_norm": None}, "Visão Brasil"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)
