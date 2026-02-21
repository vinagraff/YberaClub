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
GEO_CACHE_DIR = "/tmp/geobr_cache"


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
    os.makedirs(GEO_CACHE_DIR, exist_ok=True)
    states_cache = os.path.join(GEO_CACHE_DIR, f"states_{YEAR}.pkl")
    muni_cache = os.path.join(GEO_CACHE_DIR, f"muni_{YEAR}.pkl")
    muni_centroids_cache = os.path.join(GEO_CACHE_DIR, f"muni_centroids_{YEAR}.pkl")
    state_centroids_cache = os.path.join(GEO_CACHE_DIR, f"state_centroids_{YEAR}.pkl")

    cache_files = [states_cache, muni_cache, muni_centroids_cache, state_centroids_cache]
    if all(os.path.exists(p) for p in cache_files):
        try:
            gdf_states = pd.read_pickle(states_cache)
            gdf_muni = pd.read_pickle(muni_cache)
            muni_centroids = pd.read_pickle(muni_centroids_cache)
            state_centroids = pd.read_pickle(state_centroids_cache)
            return gdf_states, gdf_muni, muni_centroids, state_centroids
        except Exception:
            pass

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

    gdf_states.to_pickle(states_cache)
    gdf_muni.to_pickle(muni_cache)
    muni_centroids.to_pickle(muni_centroids_cache)
    state_centroids.to_pickle(state_centroids_cache)

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
state_totals = None
city_agg_by_state = None
state_meta = None
muni_by_uf = None
pts_by_uf = None
muni_geojson_by_uf = None
brazil_fig_cached = None
state_fig_cache = None


def ensure_data_loaded():
    global df, gdf_states, gdf_muni, muni_centroids, state_centroids
    global state_totals, city_agg_by_state, state_meta, muni_by_uf, pts_by_uf
    global muni_geojson_by_uf, brazil_fig_cached, state_fig_cache
    if df is None:
        df = load_data()
        gdf_states, gdf_muni, muni_centroids, state_centroids = load_geos()
        state_totals = df.groupby("estado_norm")[VALUE_COL].sum()

        city_agg = (
            df.groupby(["estado_norm", "cidade_norm"], as_index=False)[VALUE_COL]
              .sum()
        )
        city_agg_by_state = {}
        for estado_norm, grp in city_agg.groupby("estado_norm"):
            city_agg_by_state[estado_norm] = grp.rename(
                columns={"cidade_norm": "muni_name_norm"}
            )[["muni_name_norm", VALUE_COL]]

        state_meta = {}
        for row in gdf_states.itertuples():
            state_meta[row.state_name_norm] = {
                "uf": row.abbrev_state,
                "state_name": row.name_state,
            }

        muni_by_uf = {uf: grp.copy() for uf, grp in gdf_muni.groupby("abbrev_state")}
        pts_by_uf = {uf: grp.copy() for uf, grp in muni_centroids.groupby("abbrev_state")}
        muni_geojson_by_uf = {uf: with_geojson_ids(grp) for uf, grp in muni_by_uf.items()}

        brazil_fig_cached = None
        state_fig_cache = {}


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
    estado_norm: str
):
    if estado_norm not in state_meta:
        # fallback
        fig = go.Figure()
        fig.update_layout(title="Estado não encontrado no shape.")
        return fig

    uf = state_meta[estado_norm]["uf"]
    state_name = state_meta[estado_norm]["state_name"]
    agg_city = city_agg_by_state.get(estado_norm)
    if agg_city is None:
        agg_city = pd.DataFrame(columns=["muni_name_norm", VALUE_COL])

    # filtra municípios do estado e junta com os totais
    muni_state = muni_by_uf[uf].copy()
    muni_state = muni_state.merge(agg_city, on="muni_name_norm", how="left")
    muni_state[VALUE_COL] = muni_state[VALUE_COL].fillna(0)
    pts = pts_by_uf[uf].copy()

    pts = pts.merge(agg_city, on="muni_name_norm", how="left")
    pts[VALUE_COL] = pts[VALUE_COL].fillna(0)

    muni_geojson = muni_geojson_by_uf[uf]

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
        brazil_fig_cached = build_brazil_fig(df, gdf_states, state_centroids)
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
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Se clicou em voltar, volta para Brasil
    if triggered == "btn-back":
        fig = get_brazil_fig()
        return fig, {"level": "br", "estado_norm": None}, "Visão Brasil"

    # Se está na visão Brasil e clicou em um estado, faz drilldown
    if view["level"] == "br" and clickData:
        # Em mapbox, o clique pode vir do choropleth (location) ou do texto (lat/lon).
        try:
            point = clickData["points"][0]
            estado_norm = None

            if point.get("customdata"):
                # custom_data foi definido no choropleth e chega como lista/tupla.
                estado_norm = point["customdata"][0]
            elif point.get("location") is not None:
                location = str(point["location"])
                match = gdf_states[gdf_states["geo_id"] == location]
                if not match.empty:
                    estado_norm = match.iloc[0]["state_name_norm"]

            if estado_norm:
                fig = get_state_fig(estado_norm)
                return fig, {"level": "state", "estado_norm": estado_norm}, "Visão Estado (clique em Voltar para Brasil)"
        except Exception:
            pass

    # Se já está em estado, mantém estado (não faz nada no click)
    if view["level"] == "state" and view["estado_norm"]:
        fig = get_state_fig(view["estado_norm"])
        return fig, view, "Visão Estado (clique em Voltar para Brasil)"

    # default
    fig = get_brazil_fig()
    return fig, {"level": "br", "estado_norm": None}, "Visão Brasil"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)


    
