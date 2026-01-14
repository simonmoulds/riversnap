
import pandas as pd 
import geopandas as gpd

from sqlalchemy import create_engine, text, inspect

from typing import Optional
from pathlib import Path 

from riversnap.dataset.basedataset import VectorHydrographyData


__all__ = [
    "GRIT",
]


class GRIT(VectorHydrographyData):
    """GRIT vector hydrography dataset loader and candidate generator."""

    VALID_CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'SA', 'SI', 'SP']

    def __init__(self, 
                 root: Optional[Path] = None, 
                 segments: Optional[bool] = False, 
                 continents: Optional[list[str]] = None):

        """Initialize a GRIT dataset handle.

        Parameters
        ----------
        root : str or pathlib.Path
            Root directory containing GRIT GeoPackage files.
        segments : bool, optional
            If True, use segment-scale data; otherwise use reaches.
        continents : list of str, optional
            List of continent codes to search.
        """
        super().__init__(root)
        self.scale = 'segments' if segments else 'reaches'
        self.global_id = 'global_id' if segments else 'tbc'

        if continents is None: 
            self.continents = self.VALID_CONTINENTS
        else:
            if not isinstance(continents, list):
                raise ValueError('Continents must be provided as a list of continent codes.')

            if not all (c in self.VALID_CONTINENTS for c in continents):
                raise ValueError(f'Invalid continent code in {continents}. Valid codes are {self.VALID_CONTINENTS}.')

            self.continents = continents

    def load_data(self, continent: str, target_crs: int) -> gpd.GeoDataFrame:
        """Load and reproject GRIT data for a single continent.

        Parameters
        ----------
        continent : str
            Continent code.
        target_crs : int
            EPSG code to project the data to.

        Returns
        -------
        geopandas.GeoDataFrame
            Reprojected GRIT line features.
        """
        riv_shp_path =  self.root / f'GRITv1.0_{self.scale}_{continent.upper()}_EPSG4326.gpkg'
        if riv_shp_path.exists() is False:
            raise ValueError(f'GRIT data for continent "{continent}" not found at {riv_shp_path}')
        riv = gpd.read_file(riv_shp_path, layer='lines')
        riv_reproj = riv.to_crs(epsg=target_crs)
        return riv_reproj

    def get_candidates_filesystem(self, 
                                  pts: gpd.GeoDataFrame, 
                                  id_column: str, 
                                  distance_threshold: float) -> pd.DataFrame:
        candidates_list = []
        for continent in self.continents:
            riv = self.load_data(continent, target_crs=3857) # Why 3857?
            candidates = self.snap_points_to_lines(pts, riv, id_column, distance_threshold=distance_threshold)
            if candidates is not None:
                # This happens if there are no candidates within the distance threshold
                candidates_list.append(candidates)

        candidates = pd.concat(candidates_list)
        candidates = candidates.sort_values(by=[id_column, 'distance_m'])
        return candidates

    def assign_quality_flags(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Assign quality flags to candidate rows.

        Parameters
        ----------
        candidates : pandas.DataFrame
            Candidate rows to annotate.

        Returns
        -------
        pandas.DataFrame
            Candidates with a ``quality_flag`` column.
        """
        # No quality flags implemented for GRIT at present
        candidates['quality_flag'] = 0
        return candidates

    @property
    def drainage_area(self) -> pd.Series: 
        pass 