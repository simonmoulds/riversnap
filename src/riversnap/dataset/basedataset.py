
import numpy as np
import pandas as pd
import geopandas as gpd 

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

EPS = 1e-12

class HydrographyData:
    def __init__(self, root):
        self.root = Path(root)
        self.data = None
        self._candidates_cache = None
        self._candidates_cache_key = None

    def get_candidates(self): 
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError


class VectorHydrographyData(HydrographyData):

    def get_crs(self):
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
            id_column,
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

    def snap_candidates(self, 
                        pts: gpd.GeoDataFrame, 
                        id_column: str, 
                        floor_m: float = 100., 
                        w_dist: float = 0.5,
                        distance_confidence_threshold: float = 1500, 
                        pct_err_confidence_threshold: float = 25,
                        hydrorivers: bool = False,
                        return_all=False):

        """
        Snap points to lines by assessing most likely match given 
        distance and percentage error between some observed quantify 
        and its equivalent estimated quantify from the river 
        hydrography (e.g. drainage area, mean discharge).

        Parameters
        ----------
        floor_m: float 
            Minimum value of distance: values below are set to the value 
            of this parameter. The idea is to reduce the impact of
            distance values that are more precise than the resolution of 
            the dataset from which the river hydrography is derived
        w_dist: float 
            Weighting given to distance to point when scoring candidate 
            lines. 
        distance_confidence_threshold: float = 1500 
        pct_err_confidence_threshold: float = 25 
        return_all: bool 
            If True, return all candidates with computed scores and flags.
            If False, return only the best candidate per point.

        Returns
        -------
        df : DataFrame
        """
        cache_key = self._make_candidates_cache_key(
            pts=pts,
            id_column=id_column,
            distance_threshold=distance_confidence_threshold,
        )
        if self._candidates_cache_key == cache_key:
            candidates = self._candidates_cache
        else:
            candidates = self.get_candidates(pts, id_column, distance_threshold=distance_confidence_threshold)
            self._candidates_cache_key = cache_key
            self._candidates_cache = candidates

        df = candidates.copy() # To avoid modifying cached version

        # Check valid distance weighting
        if (w_dist > 1) | (w_dist < 0): 
            raise ValueError(f'w_dist must lie between 0 and 1')

        w_pct_err = 1 - w_dist
        def compute_norm_scores(df):
            # Normalize distance (per group)
            df["dist_adj"] = np.maximum(df['distance_m'].astype(float), float(floor_m))
            # df["dist_adj"] = np.minimum(df['dist_adj'].astype(float), float(ceil_m))
            df["dist_max"] = df.groupby("ohdb_id")["dist_adj"].transform("max")
            df["dist_denom"] = (df["dist_max"] - floor_m).replace(0, np.nan)
            df["dist_norm"] = (df["dist_adj"] - floor_m) / df["dist_denom"]
            df["dist_norm"] = df["dist_norm"].fillna(0.0)

            # Normalize pct_err per group
            def scale_group(x):
                return MinMaxScaler().fit_transform(x.to_numpy().reshape(-1, 1)).flatten()

            df["pct_err_norm"] = df.groupby("ohdb_id")["pct_err"].transform(scale_group)
            df["norm_score"] = w_pct_err * df["pct_err_norm"] + w_dist * df["dist_norm"]
            df = df.drop(['dist_max', 'dist_denom', 'dist_norm', 'pct_err_norm'], axis=1)
            return df

        df = compute_norm_scores(df) 

        def process_group(x):
            # Find indices of min distance and min norm
            idx_min_dist = x["distance_m"].idxmin()
            idx_min_norm = x["norm_score"].idxmin()
            if idx_min_dist == idx_min_norm:
                min_dist = x.loc[idx_min_dist, "distance_m"]
                upstream_node = x.loc[x["NEXT_DOWN"] == id, 'HYRIV_ID'].to_list()
                downstream_node = x.loc[x["HYRIV_ID"] == id, 'NEXT_DOWN'].to_list()
                nb_ids = upstream_node + downstream_node
                # Update distance_m for neighbors
                x.loc[x["HYRIV_ID"].isin(nb_ids), "distance_m"] = min_dist # Then we set equal distance to upstream/downstream nodes
            return x

        # If the candidates are drawn from HydroRIVERS then we can easily move
        # up and down the river network to assess whether the drainage area of 
        # neighbouring reaches is a better match
        if hydrorivers:
            df = df.groupby("ohdb_id", group_keys=False).apply(process_group, include_groups=True)
            df = compute_norm_scores(df) # Recompute norm scores

        # Confidence flags
        df["qflag"] = "0" # DEFAULT

        # Distance is used to assess the proximity of the station coordinates to a 
        # river reach, while pct_err is used to assess the accuracy. This could be 
        # related to drainage area, mean discharge etc. 
        # min_dist = df.groupby("ohdb_id")["dist_adj"].transform("min")
        # min_pct_err = df.groupby("ohdb_id")["pct_err"].transform("min")
        min_dist = df["dist_adj"]#.transform("min")
        min_pct_err = df["pct_err"]#.transform("min")

        # Priority order: 2.2 > 2.1 > 1 > 0
        cond_22 = (min_dist > distance_confidence_threshold) & (min_pct_err > pct_err_confidence_threshold)
        cond_21 = (min_dist <= distance_confidence_threshold) & (min_pct_err > pct_err_confidence_threshold)
        cond_1  = (min_dist > distance_confidence_threshold)

        df.loc[cond_1, "qflag"] = "1"
        df.loc[cond_21, "qflag"] = "2.1"
        df.loc[cond_22, "qflag"] = "2.2"

        # Ambiguity flag 
        df["aflag"] = "0" # DEFAULT

        # If candidates are drawn from HydroRIVERS then we can use the L12 basins
        # to assess any ambiguous cases - e.g. where the top N basins span more than 
        # one L12 basin
        if hydrorivers:
            # Compute percentage change in score
            best_score = df.groupby("ohdb_id")["norm_score"].transform("min").clip(lower=EPS)
            df['norm_score'] = df['norm_score'].clip(lower=EPS)
            df['best_score'] = best_score
            df["norm_score_pct_change"] = (df["norm_score"] - best_score).abs() / best_score

            # Keep only those scores within 10% of the best score
            df = df[df["norm_score_pct_change"] <= 0.1] # FIXME - this is a parameter

            # Flag ambiguous cases
            n_subbasins = df.groupby("ohdb_id")["HYBAS_L12"].transform("nunique")
            df.loc[n_subbasins > 1, "aflag"] = "1"
            n_basins = df.groupby("ohdb_id")["MAIN_RIV"].transform("nunique")
            df.loc[n_basins > 1, "aflag"] = "2"

        # Select best row per ohdb_id
        df = df.sort_values(["ohdb_id", "norm_score"], ascending=[True, True])

        if return_all:
            return df 

        # Otherwise we select highest scoring in each case...
        df = df.sort_values(["ohdb_id", "norm_score"], ascending=[True, True])
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
