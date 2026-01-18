
import numpy as np
import pandas as pd
import geopandas as gpd 
import pyogrio 

from sqlalchemy import create_engine, text, inspect
from pathlib import Path
from typing import List, Optional

from riversnap.utils.distance import _compute_candidate_distances_from_plan


__all__ = [
    "HydrographyData",
    "VectorHydrographyData",
    "RasterHydrographyData"
]


EPS = 1e-12


def write_points_to_postgis(pts, engine, table, if_exists='replace'): 
    pts.to_postgis(name=table, con=engine, if_exists=if_exists)
    return None 


def parse_columns(lines_columns, points_columns, lines_id_col, points_id_col, lines_geom_col, points_geom_col, duplicate_prefix="pt_"):
    """Parse and validate column names for candidate fetching."""

    if points_id_col == lines_id_col: 
        raise ValueError("points and lines must have different global ID columns names.")

    if points_id_col not in points_columns:
        raise ValueError(f"points_id_col '{points_id_col}' not found in points table columns.")

    if lines_id_col not in lines_columns:
        raise ValueError(f"lines_id_col '{lines_id_col}' not found in lines table columns.")

    columns = lines_columns + points_columns
    dupl_columns = [col for col in points_columns if col in lines_columns]
    rename_columns = {}
    if len(dupl_columns) > 0: 
        rename_columns = {col: f"{duplicate_prefix}_{col}" for col in dupl_columns if col != points_id_col}

    first_two_cols = [points_id_col, lines_id_col]
    ignore_cols = first_two_cols + [lines_geom_col, points_geom_col]
    include_columns = first_two_cols + [col for col in columns if col not in ignore_cols]
    return include_columns, rename_columns


# class PointData: # E.g. river gauges
#     def __init__(self): 
#         pass 

def get_candidates_geopackage(points, lines, points_id_col, threshold_m):        
    # Buffer points by distance
    buffered_points = points.copy()
    buffered_points["geometry"] = buffered_points.geometry.buffer(threshold_m)

    # Use spatial index + spatial join (efficient)
    # This returns all lines that intersect the buffer of any point
    candidates = gpd.sjoin(lines, buffered_points, how="inner", predicate="intersects")
    candidates = candidates.sort_values(points_id_col).reset_index(drop=True)

    # Add distances 
    ids = candidates[points_id_col].unique() 
    candidates_list = []
    for id in ids: 
        pt = points[points[points_id_col] == id]['geometry'].iloc[0]
        pt_candidates = candidates[candidates[points_id_col] == id].copy()
        pt_candidates['distance_m'] = pt_candidates.geometry.distance(pt)
        candidates_list.append(pt_candidates)

    if len(candidates_list) == 0: 
        return None

    return pd.concat(candidates_list) #[include_columns + ['distance_m']])


def get_candidates_postgis(engine=None,
                           *,
                           points: str,
                           lines: str,
                           points_geom_col: str = "geometry",
                           lines_geom_col: str = "geometry",
                           threshold_m: float = 5000.0,
                           k: int | None = 25):

    limit_sql = "" if k is None else "LIMIT :k"
    sql = f"""
    SELECT
    g.*,
    r.*,
    ST_Distance(r.{lines_geom_col}, g.{points_geom_col}) AS distance_m
    FROM {points} AS g
    JOIN LATERAL (
    SELECT *
    FROM {lines}
    WHERE ST_DWithin({lines_geom_col}, g.{points_geom_col}, :threshold_m)
    ORDER BY {lines_geom_col} <-> g.{points_geom_col}
    {limit_sql}
    ) AS r ON TRUE
    """
    # ORDER BY g.{points_id_col}, distance_m;
    params = {"threshold_m": float(threshold_m)}
    if k is not None:
        params["k"] = int(k)

    candidates = pd.read_sql(text(sql), engine, params=params)
    return candidates 


class _HydrographyBackend:
    def __init__(self): 
        pass

    def check_file_existence(self, files: List[Path] | None = None) -> bool: 
        if files is None: 
            return

        files = [Path(f) for f in files] 
        for f in files: 
            nonexistant_files = []
            if not f.exists():
                nonexistant_files.append(f)

        if len(nonexistant_files) == 1:
            raise ValueError(f"File {str(nonexistant_files[0])} does not exist!")
        if len(nonexistant_files) > 1:
            fs_string = ", ".join([str(f) for f in nonexistant_files])
            raise ValueError(f"Files {fs_string} do not exist!")

    def load_data(self, file: Path, target_crs: int, **kwargs) -> gpd.GeoDataFrame:
        """Load and reproject GRIT data for a single continent. This 
        function uses GeoPandas.read_file(...) behind the scenes.

        Parameters
        ----------
        file : Path
            File path.
        target_crs : int
            EPSG code to project the data to. 
        **kwargs : dict
            Additional keyword arguments passed to geopandas.read_file(...).

        Returns
        -------
        geopandas.GeoDataFrame
            Reprojected dataset.
        """
        if not file.exists():
            raise ValueError(f'File {file} does not exist!')

        riv = gpd.read_file(file, **kwargs)
        riv_reproj = riv.to_crs(epsg=target_crs)
        return riv_reproj

    def prepare_data_backend(self):

        raise NotImplementedError()

class _FilesystemBackend(_HydrographyBackend):
    
    def prepare_data_backend(self,
                             files=None, 
                             target_crs: int = 3857, 
                             engine=None, 
                             table: str = None, 
                             if_exists: str = "fail"): 

        self.check_file_existence(files) 
        self.lines = files
        self.srid = target_crs
        self._candidates_cache = None
        self._candidates_cache_key = None

    def make_candidates_cache_key(self, points, points_id_column, threshold_m): 
        def _hash_series(series: pd.Series) -> int:
            return int(pd.util.hash_pandas_object(series, index=False).sum())

        geom_wkb = points.geometry.apply(lambda geom: geom.wkb)
        return (
            threshold_m,
            str(points.crs),
            len(points),
            _hash_series(points[points_id_column]),
            _hash_series(geom_wkb),
        )

    def get_column_names(self, engine=None, *, gdf_or_table): 
        return list(gdf_or_table.columns) 

    def get_candidates_backend(self,
                               engine=None,
                               *,
                               points: str | gpd.GeoDataFrame, 
                               points_id_col: str,
                               points_geom_col: str = "geometry",
                               lines_geom_col: str = "geometry",
                               threshold_m: float = 5000.0,
                               k: int | None = 25):

        candidates_list = []
        for filename in self.lines: 
            lns = self.load_data(filename, target_crs=3857) # Why 3857?
            candidates = get_candidates_geopackage(points, lns, points_id_col, threshold_m)
            if candidates is not None:
                candidates_list.append(candidates)

        candidates = pd.concat(candidates_list)
        candidates = candidates.sort_values(by=[points_id_col, 'distance_m'])
        return candidates

class _PostGISBackend(_HydrographyBackend):

    def prepare_data_backend(self,
                             files=None, 
                             target_crs: int = 3857, 
                             engine: "sqlalchemy.engine.Engine" | None = None, 
                             table: str = None, 
                             if_exists: str = "fail"): 

        self.check_file_existence(files) 

        # Check existence of table
        with engine.connect() as con:
            exists = con.execute(text("SELECT to_regclass(:tname) IS NOT NULL"), {"tname": table}).scalar()

        if files is None and not exists: 
            raise ValueError("No files provided to initialize PostGIS table, and table does not already exist.")

        if not (exists and if_exists == "fail"):
            for f in files:
                ds = self.load_data(f, target_crs=target_crs)
                # ds['asset_key'] = continent
                ds.to_postgis(name=table, con=engine, if_exists=if_exists)

        self.lines = table
        self.srid = target_crs

    def make_candidates_cache_key(self, points, points_id_column, threshold_m): 
        return (points, points_id_column, threshold_m) 

    def get_column_names(self, engine, gdf_or_table): 
        sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name;
        """
        df = pd.read_sql(text(sql), engine, params={"table_name": gdf_or_table})
        return df['column_name'].tolist()

    def get_candidates_backend(self,
                               engine=None,
                               *,
                               points: str | gpd.GeoDataFrame, 
                               points_id_col: str,
                               points_geom_col: str = "geometry",
                               lines_geom_col: str = "geometry",
                               threshold_m: float = 5000.0,
                               k: int | None = 25):

        # Check if `points` table exists 
        with engine.begin() as con:
            exists = con.execute(
                text("SELECT to_regclass(:name) IS NOT NULL"), {"name": points}).scalar()

        if not exists: 
            raise ValueError(f"Table {points} does not exist in the provided PostGIS database")

        # Add index to speed things up
        with engine.begin() as con:
            con.execute(text(f"CREATE INDEX IF NOT EXISTS {self.lines}_geom_gix ON {self.lines} USING GIST (geometry);"))
            con.execute(text(f"CREATE INDEX IF NOT EXISTS {points}_geom_gix ON {points} USING GIST (geometry);"))

        candidates = get_candidates_postgis(
            engine=engine, points=points, lines=self.lines, points_geom_col=points_geom_col, 
            lines_geom_col=lines_geom_col, threshold_m=threshold_m, k=k
        )
        return candidates


class HydrographyData:
    """Base class for hydrography datasets."""
    def __init__(self, 
                 backend: str = "filesystem"): 

        """Initialize a hydrography data source.

        Parameters
        ----------
        backend : str
            The backend to use. Either "filesystem" or "postgis".
        """
        self.backend = backend
        if backend == "filesystem":
            self._backend = _FilesystemBackend() #files=..., srid=srid)
        elif backend == "postgis":
            self._backend = _PostGISBackend() #table=postgis_table, srid=srid)
        else:
            raise ValueError(...)

        self._candidates_cache_key = None 
        self._candidates_cache = None 

class VectorHydrographyData(HydrographyData):

    def prepare_data(self, **kwargs):
        self._backend.prepare_data_backend(**kwargs)

    def get_candidates(self, 
                       engine=None,
                       *,
                       points: str | gpd.GeoDataFrame, 
                       points_id_col: str,
                       points_geom_col: str = "geometry",
                       lines_geom_col: str = "geometry",
                       threshold_m: float = 5000.0,
                       k: int | None = 25, 
                       use_cache: bool = True):

        cache_key = None
        if use_cache: 
            cache_key = self._backend.make_candidates_cache_key(points=points, points_id_column=points_id_col, threshold_m=threshold_m)
            # If cache key matches, we just return the cached candidates 
            if self._candidates_cache_key == cache_key:
                return self._candidates_cache

        # Otherwise compute candidates afresh
        # lines_columns = self._backend.get_column_names(engine, gdf_or_table=self._backend.lines)
        # points_columns = self._backend.get_column_names(engine, gdf_or_table=points)
        # include_columns, rename_columns = parse_columns(lines_columns, points_columns, self.global_id, points_id_col, lines_geom_col, points_geom_col)
        candidates = self._backend.get_candidates_backend(
            engine=engine, points=points, points_id_col=points_id_col, 
            points_geom_col=points_geom_col, lines_geom_col=lines_geom_col, threshold_m=threshold_m, k=k
        )
        # candidates = candidates[include_columns + ['distance_m']] 
        # candidates = candidates.rename(columns=rename_columns)

        # Upload candidates to cache
        self._candidates_cache_key = cache_key
        self._candidates_cache = candidates

        return candidates 

    def snap(self, 
             engine=None,
             *,
             points, 
             points_id_col, 
             threshold_m, 
             threshold_m_lower: float = 100., 
             distance_specification: list = None,
             aggregation_method: str = "weighted_mean",
             return_all=False): 
        """Snap points to candidate lines and compute distances.

        Parameters
        ----------
        engine: sqlalchemy.engine.Engine
            Database engine (for PostGIS backend).
        points : geopandas.GeoDataFrame
            Point features to snap.
        points_id_column : str
            Name of the unique identifier column in ``pts``.
        threshold_m : float
            Maximum search radius in CRS units.
        threshold_m_lower : float, optional
            Minimum distance used when clipping candidate distances.
        distance_specification : list, optional
            List of distance components used for snapping.
        aggregation_method : str, optional
            Aggregation method for distance components.
        return_all : bool, optional
            If True, return all candidates with distances.

        Returns
        -------
        pandas.DataFrame
            Snapped candidates with computed distances.
        """

        lines_columns = self._backend.get_column_names(engine, gdf_or_table=self._backend.lines)
        points_columns = self._backend.get_column_names(engine, gdf_or_table=points)
        include_columns, rename_columns = parse_columns(lines_columns, points_columns, self.global_id, points_id_col)
        candidates = self.get_candidates(
            engine=engine, 
            points=points, 
            points_id_col=points_id_col, 
            points_geom_col="geometry",
            lines_geom_col="geometry",
            threshold_m=threshold_m,
            k=None, 
            use_cache=False
        )
        candidates = candidates[include_columns + ['distance_m']] 
        candidates = candidates.rename(columns=rename_columns)
        
        candidates['distance_m'] = candidates['distance_m'].clip(lower=threshold_m_lower)
        df, report = _compute_candidate_distances_from_plan(
            candidates,
            specs=distance_specification, 
            aggregation_method=aggregation_method, 
            require_any=True, 
        )
        keep_cols = (
            [points_id_col, self.global_id] 
            + report.attribute_cols 
            + report.distance_component_cols 
            + report.diagnostic_cols
            + ['distance']
        )
        df = df[keep_cols] 

        # Select best row per ohdb_id
        df = df.sort_values([points_id_col, "distance"], ascending=[True, True])
        if return_all:
            return df

        # Otherwise we select highest scoring in each case...
        df = df.groupby(points_id_col).head(1).reset_index(drop=True)
        return df
