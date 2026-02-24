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
GEO_MANIFEST_PATH = BASE_DIR / "geo_assets" / "manifest.json"

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
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    # Garante tipos e remove espaços invisíveis que quebram merge/agrupamento
    df[COL_ESTADO] = df[COL_ESTADO].astype(str).str.strip()
    df[COL_CIDADE] = df[COL_CIDADE].astype(str).str.strip()
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce").fillna(0)

    # Colunas normalizadas usadas no restante do app
    df["estado_norm"] = df[COL_ESTADO].map(norm)
    df["cidade_norm"] = df[COL_CIDADE].map(norm)

    return df


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def geojson_properties_df(geojson: dict):
    rows = []
    for feature in geojson.get("features", []):
        props = dict(feature.get("properties") or {})
        # Alguns GeoJSONs do geobr não têm feature.id; usa fallback estável.
        fid = feature.get("id")
        if fid in (None, ""):
            fid = props.get("geo_id")
        if fid in (None, ""):
            fid = props.get("code_muni")
        if fid in (None, ""):
            fid = props.get("code_state")
        props["geo_id"] = "" if fid is None else str(fid)
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


def format_br_int(value) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def build_rank_list(df_rank: pd.DataFrame, name_col: str):
    if df_rank.empty:
        return [html.Li("Sem dados", style={"color": "#64748b"})]

    items = []
    for _, row in df_rank.iterrows():
        items.append(
            html.Li(
                [
                    html.Span(str(row[name_col]), style={"fontWeight": "600", "color": "#0f172a"}),
                    html.Span(
                        format_br_int(row[VALUE_COL]),
                        style={"float": "right", "color": "#0f766e", "fontWeight": "700"},
                    ),
                ],
                style={"padding": "4px 0", "borderBottom": "1px dashed #e2e8f0"},
            )
        )
    return items


def build_rankings(view: dict):
    states_rank = (
        df.groupby(COL_ESTADO, as_index=False)[VALUE_COL]
        .sum()
        .sort_values(VALUE_COL, ascending=False)
        .head(10)
    )

    if view and view.get("level") == "state" and view.get("estado_norm"):
        df_city_base = df[df["estado_norm"] == view["estado_norm"]]
        city_title = "Top 10 Cidades (estado)"
    else:
        df_city_base = df
        city_title = "Top 10 Cidades (Brasil)"

    cities_rank = (
        df_city_base.groupby(COL_CIDADE, as_index=False)[VALUE_COL]
        .sum()
        .sort_values(VALUE_COL, ascending=False)
        .head(10)
    )

    states_children = build_rank_list(states_rank, COL_ESTADO)
    cities_children = build_rank_list(cities_rank, COL_CIDADE)
    return states_children, cities_children, city_title


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

    muni_dir = manifest.get("muni_geojson_dir") or manifest.get("municipalities_dir")
    if not muni_dir:
        raise KeyError("Manifest sem chave de diretório dos municípios (muni_geojson_dir/municipalities_dir).")

    muni_geo_path = BASE_DIR / muni_dir / f"{uf}.geojson"
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
        # garante coluna usada nos merges
        if "state_name_norm" not in states_df.columns:
            # tenta inferir a coluna de nome do estado vinda do geojson
            if "name_state" in states_df.columns:
                states_df["state_name_norm"] = states_df["name_state"].map(norm)
            elif "nome" in states_df.columns:
                states_df["state_name_norm"] = states_df["nome"].map(norm)
            elif "NAME" in states_df.columns:
                states_df["state_name_norm"] = states_df["NAME"].map(norm)
            else:
                raise KeyError(
                    f"Não encontrei coluna de nome do estado em states_df. Colunas disponíveis: {list(states_df.columns)}"
                )
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
    states_plot[VALUE_COL] = states_plot[VALUE_COL].fillna(0).astype(float)

    fig = px.choropleth(
        states_plot,
        geojson=states_geojson,
        locations="abbrev_state",
        featureidkey="properties.abbrev_state",
        color=VALUE_COL,
        custom_data=["state_name_norm"],
        color_continuous_scale="Sunsetdark",
        hover_name="name_state",
        hover_data={VALUE_COL: ":,.0f"},
        labels={VALUE_COL: "Quantidade"},
    )

    fig.update_traces(
        marker_line_width=0.8,
        marker_line_color="#F8FAFC",
        hovertemplate="<b>%{hovertext}</b><br>Quantidade: %{z:,.0f}<extra></extra>",
    )

    fig.update_layout(coloraxis_colorbar=dict(title="Quantidade", thickness=12, len=0.78))
    fig.update_geos(visible=False, fitbounds="locations", projection_type="mercator")

    # Rótulos de total por estado no próprio mapa.
    labels = (
        state_centroids.merge(
            states_plot[["abbrev_state", VALUE_COL]],
            on="abbrev_state",
            how="left",
        )
        .fillna({VALUE_COL: 0})
        .copy()
    )
    labels["label"] = (
        labels["abbrev_state"].astype(str)
        + "<br>"
        + labels[VALUE_COL].round(0).astype(int).map(lambda x: f"{x:,}".replace(",", "."))
    )

    fig.add_trace(
        go.Scattergeo(
            lat=labels["lat"],
            lon=labels["lon"],
            mode="text",
            text=labels["label"],
            textfont=dict(size=10, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    total_brasil = int(df[VALUE_COL].sum())

    fig.update_layout(
        title="Brasil - Influencers por Estado (clique para detalhar)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        annotations=[
            dict(
                x=0.01,
                y=0.02,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                borderpad=5,
                font=dict(size=12, color="#0f172a"),
                text=f"Total Brasil: {total_brasil:,}".replace(",", "."),
            )
        ],
    )

    return fig


def build_state_fig_by_uf(uf: str):
    uf = str(uf).strip().upper()

    # pega nome do estado (só pra título)
    row = states_df.loc[states_df["abbrev_state"].astype(str).str.upper() == uf]
    if row.empty:
        fig = go.Figure()
        fig.update_layout(title=f"UF não encontrada: {uf}")
        return fig

    state_name = row.iloc[0]["name_state"]

    # Se você usa muni_geojson_by_uf/pts_by_uf indexados por UF, pronto:
    if uf not in muni_geojson_by_uf or uf not in pts_by_uf:
        fig = go.Figure()
        fig.update_layout(title=f"Assets do estado não encontrados para UF: {uf}")
        return fig

    # Descobre o estado_norm correto a partir do states_df (sem risco)
    estado_norm = row.iloc[0]["state_name_norm"]

    # agrega por cidade dentro do estado (usando estado_norm correto)
    df_state = df[df["estado_norm"] == estado_norm].copy()
    agg_city = (
        df_state.groupby("cidade_norm", as_index=False)[VALUE_COL]
               .sum()
               .rename(columns={"cidade_norm": "muni_name_norm"})
    )

    pts = pts_by_uf[uf].copy()
    pts["muni_name_norm"] = pts["name_muni"].map(norm)
    pts = pts.merge(agg_city, on="muni_name_norm", how="left")
    pts[VALUE_COL] = pts[VALUE_COL].fillna(0)
    pts = pts[pts[VALUE_COL] > 0].copy()

    muni_geo = muni_geojson_by_uf[uf]
    muni_df = muni_df_by_uf[uf]
    muni_state = muni_df.copy()
    muni_state["muni_name_norm"] = muni_state["name_muni"].map(norm)
    muni_state = muni_state.merge(agg_city, on="muni_name_norm", how="left")
    muni_state[VALUE_COL] = muni_state[VALUE_COL].fillna(0).astype(float)

    fig = go.Figure()

    fig.add_trace(
        go.Choropleth(
            geojson=muni_geo,
            locations=muni_state["code_muni"],
            featureidkey="properties.code_muni",
            z=muni_state[VALUE_COL],
            colorscale="Tealgrn",
            showscale=True,
            marker_line_width=0.4,
            marker_line_color="rgba(0,0,0,0.18)",
            colorbar=dict(title="Quantidade", thickness=12, len=0.75),
            customdata=muni_state[["name_muni"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Quantidade: %{z:,.0f}<extra></extra>",
        )
    )

    top_labels = pts.nlargest(12, VALUE_COL).copy()
    top_labels["label"] = top_labels[VALUE_COL].round(0).astype(int).map(
        lambda x: f"{x:,}".replace(",", ".")
    )
    fig.add_trace(
        go.Scattergeo(
            lat=top_labels["lat"],
            lon=top_labels["lon"],
            mode="text",
            text=top_labels["label"],
            textfont=dict(size=10, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    total_state = int(df_state[VALUE_COL].sum())

    fig.update_layout(
        title=dict(
            text=f"{state_name} — Cidades (total: {total_state:,})".replace(",", "."),
            x=0.5, xanchor="center"
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=[
            dict(
                x=0.01,
                y=0.02,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                borderpad=5,
                font=dict(size=12, color="#0f172a"),
                text=f"Total do estado: {total_state:,}".replace(",", "."),
            )
        ],
        showlegend=False,
    )
    fig.update_geos(visible=False, fitbounds="locations", projection_type="mercator")

    return fig

def get_brazil_fig():
    global brazil_fig_cached
    if brazil_fig_cached is None:
        brazil_fig_cached = build_brazil_fig()
    return brazil_fig_cached


def get_state_fig(estado_norm: str):
    if not estado_norm:
        return get_brazil_fig()
    if estado_norm in state_fig_cache:
        return state_fig_cache[estado_norm]

    row = states_df.loc[states_df["state_name_norm"] == estado_norm]
    if row.empty:
        return error_figure(f"Estado não encontrado para drilldown: {estado_norm}")

    uf = str(row.iloc[0]["abbrev_state"]).upper().strip()
    load_uf_assets(uf)
    fig = build_state_fig_by_uf(uf)
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
        html.Div(
            style={"display": "flex", "gap": "14px", "alignItems": "stretch", "flexWrap": "wrap"},
            children=[
                dcc.Graph(
                    id="map",
                    style={"height": "80vh", "borderRadius": "14px", "overflow": "hidden", "flex": "1 1 760px"},
                    config={"displaylogo": False},
                ),
                html.Div(
                    style={
                        "flex": "0 0 320px",
                        "minWidth": "280px",
                        "background": "rgba(255,255,255,0.88)",
                        "border": "1px solid #cbd5e1",
                        "borderRadius": "14px",
                        "padding": "12px",
                        "boxShadow": "0 8px 20px rgba(15, 23, 42, 0.08)",
                    },
                    children=[
                        html.Div(
                            style={
                                "background": "#ffffff",
                                "border": "1px solid #dbe4ef",
                                "borderRadius": "12px",
                                "padding": "10px 12px",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Div(
                                    "Ranking Nacional",
                                    style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.4px", "color": "#64748b", "textTransform": "uppercase"},
                                ),
                                html.H4("Top 10 Estados", style={"margin": "6px 0 10px 0", "color": "#0f172a"}),
                                html.Ol(id="rank-states", style={"margin": "0 0 0 18px", "padding": 0}),
                            ],
                        ),
                        html.Div(
                            style={
                                "background": "#ffffff",
                                "border": "1px solid #dbe4ef",
                                "borderRadius": "12px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Div(
                                    "Ranking de Cidades",
                                    style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.4px", "color": "#64748b", "textTransform": "uppercase"},
                                ),
                                html.H4(
                                    id="rank-cities-title",
                                    children="Top 10 Cidades (Brasil)",
                                    style={"margin": "6px 0 10px 0", "color": "#0f172a"},
                                ),
                                html.Ol(id="rank-cities", style={"margin": "0 0 0 18px", "padding": 0}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("map", "figure"),
    Output("store-view", "data"),
    Output("subtitle", "children"),
    Output("rank-states", "children"),
    Output("rank-cities", "children"),
    Output("rank-cities-title", "children"),
    Input("map", "clickData"),
    Input("btn-back", "n_clicks"),
    State("store-view", "data"),
)
def update_map(clickData, n_back, view):
    ensure_data_loaded()

    if init_error:
        msg = f"Erro ao carregar dados geográficos: {init_error}"
        return (
            error_figure(msg),
            {"level": "br", "estado_norm": None},
            "Erro de inicialização",
            [html.Li("Erro ao carregar dados")],
            [html.Li("Erro ao carregar dados")],
            "Top 10 Cidades",
        )

    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "btn-back":
        new_view = {"level": "br", "estado_norm": None}
        rs, rc, ctitle = build_rankings(new_view)
        return get_brazil_fig(), new_view, "Visão Brasil", rs, rc, ctitle

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
                new_view = {"level": "state", "estado_norm": estado_norm}
                rs, rc, ctitle = build_rankings(new_view)
                return fig, new_view, "Visão Estado (clique em Voltar para Brasil)", rs, rc, ctitle
        except Exception:
            pass

    if view["level"] == "state" and view["estado_norm"]:
        fig = get_state_fig(view["estado_norm"])
        rs, rc, ctitle = build_rankings(view)
        return fig, view, "Visão Estado (clique em Voltar para Brasil)", rs, rc, ctitle

    new_view = {"level": "br", "estado_norm": None}
    rs, rc, ctitle = build_rankings(new_view)
    return get_brazil_fig(), new_view, "Visão Brasil", rs, rc, ctitle


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)
