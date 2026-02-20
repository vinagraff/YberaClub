import pandas as pd
import geopandas as gpd
from unidecode import unidecode

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


def build_brazil_fig(df, gdf_states, state_centroids):
    # agrega por estado (nome no excel) e junta com geobr por nome normalizado
    agg_state = (
        df.groupby("estado_norm", as_index=False)[VALUE_COL]
          .sum()
          .rename(columns={"estado_norm": "state_name_norm"})
    )

    states_plot = gdf_states.merge(agg_state, on="state_name_norm", how="left")
    states_plot[VALUE_COL] = states_plot[VALUE_COL].fillna(0)

    # choropleth dos estados
    fig = px.choropleth(
        states_plot,
        geojson=states_plot.geometry.__geo_interface__,
        locations=states_plot.index,
        color=VALUE_COL,
        color_continuous_scale="YlOrRd",
        hover_name="name_state",
        hover_data={VALUE_COL: ":,.0f"},
        labels={VALUE_COL: "Quantidade"},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )

    # texto com totais em cima de cada estado (centróides)
    txt = state_centroids.merge(
        agg_state, on="state_name_norm", how="left"
    )
    txt[VALUE_COL] = txt[VALUE_COL].fillna(0)

    fig.add_trace(
        go.Scattergeo(
            lat=txt["lat"],
            lon=txt["lon"],
            text=txt[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", ".")),
            mode="text",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Brasil — Quantidade por Estado (clique em um estado para detalhar)",
        margin=dict(l=10, r=10, t=60, b=10),
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
    pts = muni_centroids[muni_centroids["abbrev_state"] == uf].copy()

    pts = pts.merge(agg_city, on="muni_name_norm", how="left")
    pts[VALUE_COL] = pts[VALUE_COL].fillna(0)

    # Base: polígono do estado (para dar contexto) + zoom
    fig = px.choropleth(
        muni_state,
        geojson=muni_state.geometry.__geo_interface__,
        locations=muni_state.index,
        color_discrete_sequence=["#EAEAEA"],  # só para desenhar os limites
    )

    fig.update_traces(marker_line_width=0.5, marker_line_color="white", hoverinfo="skip")

    # pontos das cidades com tamanho proporcional
    # (só plota onde tem valor > 0)
    pts_plot = pts[pts[VALUE_COL] > 0].copy()

    fig.add_trace(
        go.Scattergeo(
            lat=pts_plot["lat"],
            lon=pts_plot["lon"],
            text=pts_plot["name_muni"] + "<br>" + pts_plot[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", ".")),
            hovertemplate="%{text}<extra></extra>",
            mode="markers",
            marker=dict(
                size=(pts_plot[VALUE_COL] ** 0.5) * 2.5,  # escala visual
                opacity=0.75,
            ),
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )

    total_state = int(df_state[VALUE_COL].sum())

    fig.update_layout(
        title=f"{state_name} — Cidades (total no estado: {total_state:,})".replace(",", "."),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


# -----------------------------
# App
# -----------------------------
df = load_data()
gdf_states, gdf_muni, muni_centroids, state_centroids = load_geos()

app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"},
    children=[
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center"},
            children=[
                html.Button("Voltar para Brasil", id="btn-back", n_clicks=0),
                html.Div(id="subtitle", style={"color": "#444"}),
            ],
        ),
        dcc.Store(id="store-view", data={"level": "br", "estado_norm": None}),
        dcc.Graph(id="map", style={"height": "80vh"}),
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
            idx = clickData["points"][0]["location"]
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
    app.run(debug=True)