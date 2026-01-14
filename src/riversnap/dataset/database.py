
import pandas as pd
import geopandas as gpd 

from sqlalchemy import create_engine, text

# What I did to get to this point: 
# ================================
# 1 - Installed Docker Desktop
# 2 - Started a containerized postgis instance locally:
# > docker run --name postgis \
#     -e POSTGRES_PASSWORD=postgres \
#     -e POSTGRES_USER=postgres \
#     -e POSTGRES_DB=riversnap \
#     -p 5432:5432 \
#     -d postgis/postgis:16-3.4
# 3 - Connected to the database [not necessary]
# docker exec -ti postgis psql "postgresql://postgres:postgres@localhost:5432/riversnap"

# engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/riversnap")
# ds = gpd.read_file('/Users/smoulds/data/GRITv1/GRITv1.0_segments_EU_EPSG4326.gpkg')
# ds = ds.to_crs(3857)
# ds.to_postgis(name='gritv1_segments', con=engine, if_exists='replace')
# pts = gpd.read_file('/Users/smoulds/data/ohdb_stations.gpkg')
# pts = pts[pts['ohdb_source']=='NRFA'] # Restrict to UK gauges for now 
# pts['ohdb_source_id'] = pts['ohdb_source_id'].astype(float).astype(int)
# pts = pts[['ohdb_id', 'ohdb_source', 'ohdb_source_id', 'ohdb_catchment_area', 'geometry']] # Keep relevant columns only
# pts_reproj = pts.to_crs(epsg=3857)
# pts_reproj.to_postgis(name='ohdb_nrfa', con=engine, if_exists='replace')
# with engine.begin() as con:
#     con.execute(text("CREATE INDEX IF NOT EXISTS gritv1_segments_geom_gix ON gritv1_segments USING GIST (geometry);"))
#     con.execute(text("CREATE INDEX IF NOT EXISTS ohdb_nrfa_geom_gix ON ohdb_nrfa USING GIST (geometry);"))

def write_points_to_postgis(pts, engine, table, if_exists='replace'): 
    pts.to_postgis(name=table, con=engine, if_exists=if_exists)
    return None 


def fetch_candidates_topk_geom(
    engine,
    *,
    points_table: str,
    lines_table: str,
    gauge_id_col: str,
    reach_id_col: str,
    geom_col_points: str = "geom",
    geom_col_lines: str = "geom",
    threshold_m: float = 5000.0,
    k: int | None = 25,
) -> pd.DataFrame:

    limit_sql = "" if k is None else "LIMIT: k"

    sql = f"""
    SELECT
      g.{gauge_id_col} AS gauge_id,
      r.{reach_id_col} AS reach_id,
      ST_Distance(r.{geom_col_lines}, g.{geom_col_points}) AS distance_m
    FROM {points_table} AS g
    JOIN LATERAL (
      SELECT {reach_id_col}, {geom_col_lines}
      FROM {lines_table}
      WHERE ST_DWithin({geom_col_lines}, g.{geom_col_points}, :threshold_m)
      ORDER BY {geom_col_lines} <-> g.{geom_col_points}
      {limit_sql}
    ) AS r
    ON TRUE
    ORDER BY g.{gauge_id_col}, distance_m;
    """
    
    params={"threshold_m": float(threshold_m)} 
    if k is not None: 
        params["k"] = int(k)

    return pd.read_sql(text(sql), engine, params=params)

# cand = fetch_candidates_topk_geom(
#     engine,
#     points_table="ohdb_nrfa",
#     lines_table="gritv1_segments",
#     gauge_id_col='ohdb_id',
#     reach_id_col='global_id',
#     geom_col_points='geometry',
#     geom_col_lines='geometry',
#     threshold_m=5000,
#     k=25
# )
