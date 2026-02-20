import pandas as pd
import geopandas as gpd
from unidecode import unidecode
import json
import os

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context
import geobr


# -----------------------------
# Config
# -----------------------------
EXCEL_PATH = "AfiliadosAtivos_EstadoCidade.xlsx"  # ajuste se precisar
VALUE_COL = "qt_influencers"
COL_ESTADO = "estado"
COL_CIDADE = "cidade"

YEAR = 2020  # ano dos shapes (IBGE)
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
    # normalizações para join
    df["estado_norm"] = df[COL_ESTADO].map(norm)
    df["cidade_norm"] = df[COL_CIDADE].map(norm)
    return df


def load_geos():
    # Estados (UF)
    gdf_states = geobr.read_state(year=YEAR)
    gdf_states = gdf_states.to_crs(4674)  # lat/lon

    # Municípios
    gdf_muni = geobr.read_municipality(year=YEAR)
    gdf_muni = gdf_muni.to_crs(4674)

    # normalizações para join com seu excel
    gdf_states["state_name_norm"] = gdf_states["name_state"].map(norm)
    gdf_muni["muni_name_norm"] = gdf_muni["name_muni"].map(norm)
    gdf_states["geo_id"] = gdf_states.index.astype(str)
    gdf_muni["geo_id"] = gdf_muni.index.astype(str)

    # centróides (para colocar os pontos das cidades)
    muni_centroids = gdf_muni.copy()
    muni_centroids["geometry"] = muni_centroids.geometry.centroid
    muni_centroids["lat"] = muni_centroids.geometry.y
    muni_centroids["lon"] = muni_centroids.geometry.x

    # centróides dos estados (para escrever o total em cima do mapa)
    state_centroids = gdf_states.copy()
    state_centroids["geometry"] = state_centroids.geometry.centroid
    state_centroids["lat"] = state_centroids.geometry.y
    state_centroids["lon"] = state_centroids.geometry.x

    return gdf_states, gdf_muni, muni_centroids, state_centroids


def with_geojson_ids(gdf: gpd.GeoDataFrame):
    geojson = json.loads(gdf.to_json())
    for feature, geo_id in zip(geojson["features"], gdf["geo_id"].tolist()):
        feature["id"] = str(geo_id)
    return geojson


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


# lazy load para não bloquear o boot do gunicorn antes de bindar a porta
df = None
gdf_states = None
gdf_muni = None
muni_centroids = None
state_centroids = None


def ensure_data_loaded():
    global df, gdf_states, gdf_muni, muni_centroids, state_centroids
    if df is None:
        df = load_data()
        gdf_states, gdf_muni, muni_centroids, state_centroids = load_geos()


def build_brazil_fig(df, gdf_states, state_centroids):
    # agrega por estado (nome no excel) e junta com geobr por nome normalizado
    agg_state = (
        df.groupby("estado_norm", as_index=False)[VALUE_COL]
          .sum()
          .rename(columns={"estado_norm": "state_name_norm"})
    )

    states_plot = gdf_states.merge(agg_state, on="state_name_norm", how="left")
    states_plot[VALUE_COL] = states_plot[VALUE_COL].fillna(0)

    states_geojson = with_geojson_ids(states_plot)

    fig = px.choropleth_mapbox(
        states_plot,
        geojson=states_geojson,
        locations="geo_id",
        featureidkey="id",
        color=VALUE_COL,
        color_continuous_scale="Sunsetdark",
        hover_name="name_state",
        hover_data={VALUE_COL: ":,.0f"},
        labels={VALUE_COL: "Quantidade"},
        opacity=0.88,
        mapbox_style=MAP_STYLE,
        center={"lat": -14.2, "lon": -51.9},
        zoom=3.4,
    )

    fig.update_layout(
        mapbox=dict(pitch=42, bearing=-14),
        coloraxis_colorbar=dict(
            title="Quantidade",
            thickness=12,
            len=0.78,
        ),
    )

    # texto com totais em cima de cada estado (centróides)
    txt = state_centroids.merge(
        agg_state, on="state_name_norm", how="left"
    )
    txt[VALUE_COL] = txt[VALUE_COL].fillna(0)

    fig.add_trace(
        go.Scattermapbox(
            lat=txt["lat"],
            lon=txt["lon"],
            text=txt[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", ".")),
            mode="text",
            hoverinfo="skip",
            textfont=dict(size=13, color="#1A2333"),
        )
    )

    fig.update_layout(
        title="Brasil - Influencers por Estado (clique para detalhar)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
    )

    return fig


def build_state_fig(
    df, gdf_states, gdf_muni, muni_centroids, estado_norm: str
):
    # identifica a UF no geobr
    row_state = gdf_states.loc[gdf_states["state_name_norm"] == estado_norm]
    if row_state.empty:
        # fallback
        fig = go.Figure()
        fig.update_layout(title="Estado não encontrado no shape.")
        return fig

    uf = row_state.iloc[0]["abbrev_state"]
    state_name = row_state.iloc[0]["name_state"]

    # agrega por cidade (dentro do estado)
    df_state = df[df["estado_norm"] == estado_norm].copy()
    agg_city = (
        df_state.groupby("cidade_norm", as_index=False)[VALUE_COL]
               .sum()
               .rename(columns={"cidade_norm": "muni_name_norm"})
    )

    # filtra municípios do estado e junta com os totais
    muni_state = gdf_muni[gdf_muni["abbrev_state"] == uf].copy()
    muni_state = muni_state.merge(agg_city, on="muni_name_norm", how="left")
    muni_state[VALUE_COL] = muni_state[VALUE_COL].fillna(0)
    pts = muni_centroids[muni_centroids["abbrev_state"] == uf].copy()

    pts = pts.merge(agg_city, on="muni_name_norm", how="left")
    pts[VALUE_COL] = pts[VALUE_COL].fillna(0)

    muni_geojson = with_geojson_ids(muni_state)

    fig = px.choropleth_mapbox(
        muni_state,
        geojson=muni_geojson,
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

    # pontos das cidades com tamanho proporcional
    # (só plota onde tem valor > 0)
    pts_plot = pts[pts[VALUE_COL] > 0].copy()

    # camada de sombra para dar profundidade nos pontos
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
            text=pts_plot["name_muni"] + "<br>" + pts_plot[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", ".")),
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

    bounds = muni_state.total_bounds
    center = {"lat": (bounds[1] + bounds[3]) / 2, "lon": (bounds[0] + bounds[2]) / 2}
    zoom = fit_zoom(bounds)
    fig.update_layout(
        mapbox=dict(center=center, zoom=zoom, pitch=50, bearing=18),
        coloraxis_colorbar=dict(title="Qtd", thickness=10, len=0.7),
    )

    total_state = int(df_state[VALUE_COL].sum())

    fig.update_layout(
        title=f"{state_name} - Cidades (total no estado: {total_state:,})".replace(",", "."),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        showlegend=False,
    )
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
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Se clicou em voltar, volta para Brasil
    if triggered == "btn-back":
        fig = build_brazil_fig(df, gdf_states, state_centroids)
        return fig, {"level": "br", "estado_norm": None}, "Visão Brasil"

    # Se está na visão Brasil e clicou em um estado, faz drilldown
    if view["level"] == "br" and clickData:
        # hover_name vem do 'name_state'. Vamos extrair o estado clicado:
        # O clickData do choropleth terá pointNumber = index do geodataframe usado.
        try:
            location = clickData["points"][0]["location"]
            idx = int(location)
            estado_norm = gdf_states.iloc[idx]["state_name_norm"]
            fig = build_state_fig(df, gdf_states, gdf_muni, muni_centroids, estado_norm)
            return fig, {"level": "state", "estado_norm": estado_norm}, "Visão Estado (clique em Voltar para Brasil)"
        except Exception:
            pass

    # Se já está em estado, mantém estado (não faz nada no click)
    if view["level"] == "state" and view["estado_norm"]:
        fig = build_state_fig(df, gdf_states, gdf_muni, muni_centroids, view["estado_norm"])
        return fig, view, "Visão Estado (clique em Voltar para Brasil)"

    # default
    fig = build_brazil_fig(df, gdf_states, state_centroids)
    return fig, {"level": "br", "estado_norm": None}, "Visão Brasil"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)


    
