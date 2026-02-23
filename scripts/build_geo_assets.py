from pathlib import Path
import json

import pandas as pd
import geopandas as gpd
import geobr


YEAR = 2020

# Raiz do projeto (scripts/..)
BASE_DIR = Path(__file__).resolve().parents[1]

# 👉 IMPORTANTE:
# Ajuste a pasta abaixo para bater com o que o seu main.py define em GEO_MANIFEST_PATH
# Exemplo comum: "geo_assets"
OUT_DIR = BASE_DIR / "geo_assets"

MANIFEST_NAME = "manifest.json"


def centroids_latlon(gdf_latlon: gpd.GeoDataFrame, projected_epsg: int = 3857) -> pd.DataFrame:
    """Calcula centróides corretamente (projeta em metros, calcula, volta para lat/lon)."""
    gdf_proj = gdf_latlon.to_crs(epsg=projected_epsg)
    cent = gdf_proj.geometry.centroid
    cent_latlon = gpd.GeoSeries(cent, crs=gdf_proj.crs).to_crs(gdf_latlon.crs)

    return pd.DataFrame({"lon": cent_latlon.x, "lat": cent_latlon.y})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    muni_geojson_dir = OUT_DIR / "muni_geojson"
    centroids_dir = OUT_DIR / "centroids"
    muni_geojson_dir.mkdir(parents=True, exist_ok=True)
    centroids_dir.mkdir(parents=True, exist_ok=True)

    print("Baixando shapes (IBGE) via geobr...")
    states = geobr.read_state(year=YEAR).to_crs(4674)
    muni = geobr.read_municipality(year=YEAR).to_crs(4674)

    # --- States geojson ---
    states_geo_path = OUT_DIR / "states.geojson"
    print(f"Salvando: {states_geo_path}")
    states.to_file(states_geo_path, driver="GeoJSON")

    # --- State centroids ---
    sc = centroids_latlon(states)
    state_centroids = pd.DataFrame({
        "abbrev_state": states["abbrev_state"].values,
        "name_state": states["name_state"].values,
        "lat": sc["lat"].values,
        "lon": sc["lon"].values,
    })
    state_centroids_path = OUT_DIR / "state_centroids.csv"
    print(f"Salvando: {state_centroids_path}")
    state_centroids.to_csv(state_centroids_path, index=False)

    # --- Muni geojson + centroids por UF ---
    ufs = sorted(muni["abbrev_state"].unique().tolist())
    for uf in ufs:
        muni_uf = muni[muni["abbrev_state"] == uf].copy()

        geo_path = muni_geojson_dir / f"{uf}.geojson"
        print(f"[{uf}] Salvando muni geojson: {geo_path}")
        muni_uf.to_file(geo_path, driver="GeoJSON")

        c = centroids_latlon(muni_uf)
        pts = pd.DataFrame({
            "name_muni": muni_uf["name_muni"].values,
            "code_muni": muni_uf["code_muni"].values,
            "lat": c["lat"].values,
            "lon": c["lon"].values,
        })

        pts_path = centroids_dir / f"{uf}_centroids.csv"
        print(f"[{uf}] Salvando centróides: {pts_path}")
        pts.to_csv(pts_path, index=False)

    # --- Manifest ---
    manifest = {
        "states_geojson": str(states_geo_path.relative_to(BASE_DIR)),
        "state_centroids_csv": str(state_centroids_path.relative_to(BASE_DIR)),
        "muni_geojson_dir": str(muni_geojson_dir.relative_to(BASE_DIR)),
        "centroids_dir": str(centroids_dir.relative_to(BASE_DIR)),
        "state_meta": {uf: {} for uf in ufs},  # opcional; pode ficar vazio
    }

    manifest_path = OUT_DIR / MANIFEST_NAME
    print(f"Salvando manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ OK: assets e manifest gerados.")


if __name__ == "__main__":
    main()