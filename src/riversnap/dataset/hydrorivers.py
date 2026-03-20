
from typing import Optional, List
from pathlib import Path 

from riversnap.dataset.basedataset import VectorHydrographyData


__all__ = [
    "HydroRIVERS",
]


class HydroRIVERS(VectorHydrographyData): 

    VALID_CONTINENTS = []

    def get_files(self, root) -> list[Path]:
        """Get list of data files for specified continents.

        Parameters
        ----------
        root : pathlib.Path
            Root directory containing GRIT GeoPackage files.

        Returns
        -------
        list of pathlib.Path
            List of GRIT data file paths.
        """
        riv_shp_path =  root / 'RiverATLAS_v10.gdb'
        if riv_shp_path.exists() is False:
            raise ValueError(f'Data not found at {riv_shp_path}')
        return [riv_shp_path]
