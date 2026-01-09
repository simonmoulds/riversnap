
import numpy as np
import pandas as pd
import geopandas as gpd 

from pathlib import Path
from riversnap.utils.distance import compute_candidate_distances_from_plan

EPS = 1e-12

class HydrographyData:
    """Base class for hydrography data sources."""
    def __init__(self, root):
        """Initialize a hydrography data source.

        Parameters
        ----------
        root : str or pathlib.Path
            Root directory containing hydrography data files.
        """
        self.root = Path(root)
        self.data = None
        self._candidates_cache = None
        self._candidates_cache_key = None

    def get_candidates(self): 
        """Return candidate line features for the given points.

        Returns
        -------
        pandas.DataFrame
            Candidate features for each point.
        """
        raise NotImplementedError

    def load_data(self):
        """Load source data from disk or a remote store.

        Returns
        -------
        object
            Loaded dataset object.
        """
        raise NotImplementedError


class VectorHydrographyData(HydrographyData):
    """Base class for vector-based hydrography datasets."""
    def _get_crs(self):
        if self.data is not None:
            return self.data.crs
        else:
            raise ValueError("Data not loaded. Please call load_data() first.")

    def _make_candidates_cache_key(
        self,
        pts: gpd.GeoDataFrame,
        id_column: str,
        distance_threshold: float,
    ) -> tuple:
        def _hash_series(series: pd.Series) -> int:
            return int(pd.util.hash_pandas_object(series, index=False).sum())

        geom_wkb = pts.geometry.apply(lambda geom: geom.wkb)
        return (
            distance_threshold,
            str(pts.crs),
            len(pts),
            _hash_series(pts[id_column]),
            _hash_series(geom_wkb),
        )
        
    def snap_points_to_lines(self, pts, lns, id_column, distance_threshold=1500): 
        """
        Snap points [e.g. gauging stations] to lines [e.g. river reaches]
        within a given distance threshold.

        Parameters
        ----------
        pts : GeoDataFrame
            Input points to be snapped.
        lns : GeoDataFrame
            Input lines to snap points to.
        id_column : str
            Name of the column in `pts` that contains unique point IDs.
        distance_threshold : float, optional
            Maximum snapping distance in the units of the coordinate
            reference system (CRS). Lines farther than this distance from
            any given point will be discarded. Default is 1500.

        Returns
        -------
        candidates : GeoDataFrame
            A list of candidate lines for each point.

        Notes
        -----
        - If `pts` or `lns` are GeoDataFrames, both must be in the same CRS.
        - The snapping is performed geometrically, not topologically.
        """
        if id_column not in pts.columns:
            raise ValueError(f'ID column "{id_column}" not found in points GeoDataFrame.')

        # Step 1: Buffer points by distance
        buffered_points = pts.copy()
        buffered_points["geometry"] = buffered_points.geometry.buffer(distance_threshold)

        # Step 2: Use spatial index + spatial join (efficient)
        # This returns all lines that intersect the buffer of any point
        candidates = gpd.sjoin(lns, buffered_points, how="inner", predicate="intersects")
        candidates = candidates.sort_values(id_column).reset_index(drop=True)

        # Step 3: Add distances 
        ids = candidates[id_column].unique() 
        candidates_list = []
        for id in ids: 
            pt = pts[pts[id_column] == id]['geometry'].iloc[0]
            pt_candidates = candidates[candidates[id_column] == id].copy()
            pt_candidates['distance_m'] = pt_candidates.geometry.distance(pt)
            candidates_list.append(pt_candidates)

        if len(candidates_list) > 0: 
            return pd.concat(candidates_list)
        else:
            return None

    def _get_candidates_from_cache(self, pts, id_column, distance_threshold): 
        cache_key = self._make_candidates_cache_key(
            pts=pts,
            id_column=id_column,
            distance_threshold=distance_threshold,
        )
        if self._candidates_cache_key == cache_key:
            candidates = self._candidates_cache
        else:
            candidates = self.get_candidates(pts, id_column, distance_threshold=distance_threshold)
            self._candidates_cache_key = cache_key
            self._candidates_cache = candidates
        return candidates

    def snap(self, 
             pts, 
             id_column, 
             distance_threshold, 
             distance_lower_threshold: float = 100., 
             distance_specification: list = None,
             aggregation_method: str = "weighted_mean",
             return_all=False): 
        """Snap points to candidate lines and compute distances.

        Parameters
        ----------
        pts : geopandas.GeoDataFrame
            Point features to snap.
        id_column : str
            Name of the unique identifier column in ``pts``.
        distance_threshold : float
            Maximum search radius in CRS units.
        distance_lower_threshold : float, optional
            Minimum distance used when clipping candidate distances.
        distance_plan : list, optional
            Distance plan passed to ``compute_candidate_distances_from_plan``.
        aggregation_method : str, optional
            Aggregation method for component distances.
        return_all : bool, optional
            If True, return all candidates with distances.

        Returns
        -------
        pandas.DataFrame
            Snapped candidates with computed distances.
        """

        candidates = self._get_candidates_from_cache(pts, id_column, distance_threshold)

        # keep_cols = [self.global_id, id_column, 'distance_m']
        # candidates = candidates[keep_cols]

        candidates['distance_m'] = candidates['distance_m'].clip(lower=distance_lower_threshold)
        df = compute_candidate_distances_from_plan(
            candidates,
            specs=distance_specification, 
            aggregation_method=aggregation_method, 
            require_any=True, 
        )

        # Select best row per ohdb_id
        df = df.sort_values([id_column, "distance"], ascending=[True, True])
        if return_all:
            return df 

        # Otherwise we select highest scoring in each case...
        df = df.groupby("ohdb_id").head(1).reset_index(drop=True)
        return df 


# class RasterHydrographyData(HydrographyData):
#     def get_bounds(self):
#         if self.data is not None:
#             return self.data.total_bounds
#         else:
#             raise ValueError("Data not loaded. Please call load_data() first.")
        
#     def snap_points_to_accumulation_raster(self, 
#                                            pts, 
#                                            raster_path, 
#                                            accumulated_area_scale_factor=1.,
#                                            global_accumulated_area_threshold=None,
#                                            catchment_accumulated_area_error=None, 
#                                            catchment_area_column=None,
#                                            distance_threshold=1500,
#                                            n_candidates=None):
#         """
#         Snap points (e.g., gauges) to the pixel of maximum accumulated area
#         within a given search radius.

#         Parameters
#         ----------
#         pts : GeoDataFrame
#             Input point locations. Must have the same CRS as the raster.
#         raster_path : str
#             Path to a flow accumulation raster (e.g., upstream area).
#         accumulated_area_scale_factor: float 
#             Scale factor to apply to accumulated area map 
#         global_accumulated_area_threshold: float 
#             Threshold below which accumulated area values are masked. 

#         catchment_accumulated_area_threhsold: float 
#         catchment_area_column: str
#         distance_threshold : float
#             Search radius for snapping, in CRS units (typically metres).
#         n_candidates: int or None
#             Number of candidate pixels to return per point.
#             If None, return all valid candidates.

#         Returns
#         -------
#         snapped : GeoDataFrame
#             A GeoDataFrame containing the new snapped point location, 
#             original point id, raster value, and distance to snapped location.
#         """

#         results = []
#         with rasterio.open(raster_path) as src:
#             for _, row in tqdm(pts.iterrows(), total=len(pts)):
#                 pt = row.geometry
#                 pid = row.get("gauge_id", _)

#                 # Bounding box around point
#                 minx, maxx = pt.x - distance_threshold, pt.x + distance_threshold
#                 miny, maxy = pt.y - distance_threshold, pt.y + distance_threshold

#                 # Raster window
#                 window = from_bounds(minx, miny, maxx, maxy, src.transform)
#                 arr = src.read(1, window=window, masked=True)
#                 arr = arr * accumulated_area_scale_factor # Multiply by scale factor

#                 if arr.mask.all():
#                     continue

#                 # Apply circular mask
#                 height, width = arr.shape
#                 xs = np.arange(width) * src.transform.a + minx + src.transform.a/2
#                 ys = np.arange(height) * src.transform.e + maxy + src.transform.e/2
#                 X, Y = np.meshgrid(xs, ys)
#                 dist2 = np.sqrt((X - pt.x)**2 + (Y - pt.y)**2)

#                 if catchment_accumulated_area_error is not None: 
#                     if catchment_area_column is None or catchment_area_column not in row: 
#                         raise ValueError()
#                     area = float(row[catchment_area_column])
#                     area_err = 100 * (np.abs(arr.data - area) / np.clip(np.abs(area), EPS, None))
#                     area_mask = area_err < catchment_accumulated_area_error

#                 elif global_accumulated_area_threshold is not None: 
#                     area_mask = (arr.data >= global_accumulated_area_threshold)
#                 else:
#                     area_mask = (arr.data >= arr.min())

#                 valid_mask = (~arr.mask) & (dist2 <= distance_threshold) & area_mask

#                 if not valid_mask.any():
#                     continue

#                 # Get indices and values of valid pixels
#                 valid_indices = np.array(np.where(valid_mask)).T  # (row, col)
#                 valid_values = dist2[valid_mask]

#                 # Sort by accumulated area descending
#                 sorted_idx = np.argsort(valid_values)

#                 if n_candidates is not None:
#                     top_idx = sorted_idx[:n_candidates]
#                 else:
#                     top_idx = sorted_idx

#                 pt_results = []
#                 for i in top_idx:
#                     r, c = valid_indices[i]
#                     x_snap, y_snap = X[r, c], Y[r, c]
#                     snap_pt = Point(x_snap, y_snap)
#                     pt_results.append({
#                         "gauge_id": pid,
#                         "geometry": snap_pt,
#                         "accum_area": arr.data[r, c],
#                         "distance_m": pt.distance(snap_pt)
#                     })

#                 pt_results = pd.DataFrame(pt_results)
#                 results.append(pt_results)

#         return gpd.GeoDataFrame(
#             pd.concat(results),
#             geometry='geometry',
#             crs=pts.crs
#         )
