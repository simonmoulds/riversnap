
from typing import Optional, List
from pathlib import Path 

from riversnap.dataset.basedataset import VectorHydrographyData


__all__ = [
    "MERIT_Hydro_v07_Basins",
]


class MERIT_Hydro_v07_Basins(VectorHydrographyData): 
    """MERIT Hydro v07 Basins hydrography dataset loader."""

    VALID_CONTINENTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_files(self, root) -> list[Path]:
        """Get list of data files for specified continents.

        Parameters
        ----------
        root : pathlib.Path
            Root directory containing GRIT GeoPackage files.
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
            riv_shp_path =  root / 'pfaf_level_01' / f'riv_pfaf_{continent}_MERIT_Hydro_v07_Basins_v01.shp'
            if riv_shp_path.exists() is False:
                raise ValueError(f'Data for continent "{continent}" not found at {riv_shp_path}')
            files.append(riv_shp_path)
        return files