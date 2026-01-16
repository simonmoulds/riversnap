
import pandas as pd 
import geopandas as gpd

from sqlalchemy import create_engine, text, inspect

from typing import Optional, List
from pathlib import Path 

from riversnap.dataset.basedataset import VectorHydrographyData


__all__ = [
    "GRIT",
]


GRIT_CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'SA', 'SI', 'SP']

class GRIT(VectorHydrographyData):
    """GRIT vector hydrography dataset loader and candidate generator."""

    def __init__(self, 
                 backend: str,
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
        super().__init__(backend=backend)

        self.scale = 'segments' if segments else 'reaches'
        self.global_id = 'global_id' if segments else 'tbc'

        if continents is None: 
            self.continents = GRIT_CONTINENTS
        else:
            if not isinstance(continents, list):
                raise ValueError('Continents must be provided as a list of continent codes.')

            if not all (c in GRIT_CONTINENTS for c in continents):
                raise ValueError(f'Invalid continent code in {continents}. Valid codes are {GRIT_CONTINENTS}.')
            
            self.continents = continents

    def get_files(self, root, continents: List[str] = GRIT_CONTINENTS, grit_version=1, srid=4326) -> list[Path]:
        """Get list of GRIT data files for specified continents.

        Parameters
        ----------
        root : pathlib.Path
            Root directory containing GRIT GeoPackage files.
        continents : list of str
            List of continent codes to search.
        grit_version : int 
            GRIT version number (default is 1).
        srid : int 
            Coordinate reference system EPSG code (default is 4326).

        Returns
        -------
        list of pathlib.Path
            List of GRIT data file paths.
        """
        files = []
        for continent in continents:
            riv_shp_path =  root / f'GRITv1.0_{self.scale}_{continent.upper()}_EPSG4326.gpkg'
            if riv_shp_path.exists() is False:
                raise ValueError(f'GRIT data for continent "{continent}" not found at {riv_shp_path}')
            files.append(riv_shp_path)
        return files