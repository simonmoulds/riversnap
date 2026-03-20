
from typing import Optional, List
from pathlib import Path 

from riversnap.dataset.basedataset import VectorHydrographyData


__all__ = [
    "GRIT",
]

# Separate class for segments and reaches? 

class GRIT(VectorHydrographyData):
    """GRIT vector hydrography dataset loader and candidate generator."""

    VALID_CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'SA', 'SI', 'SP']

    def __init__(self, 
                 backend: str,
                 continents: list[str] | None = None,
                 segments: bool = False):

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

    def get_files(self, 
                  root: Path, 
                  srid: int = 4326) -> list[Path]:

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
        # FIXME this could return a dict with the source continent
        files = []
        for continent in self.continents:
            riv_shp_path =  root / f'GRITv1.0_{self.scale}_{continent.upper()}_EPSG{srid}.gpkg'
            if riv_shp_path.exists() is False:
                raise ValueError(f'Data for continent "{continent}" not found at {riv_shp_path}')
            files.append(riv_shp_path)
        return files