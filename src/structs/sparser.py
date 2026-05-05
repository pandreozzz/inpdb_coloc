from __future__ import annotations

from typing import Union, Optional, List
import pandas as pd
import numpy as np
import xarray as xr

from .cores import SparseIndex

class INPIndex:
    """Helper class to store index arrays for all attributes"""
    def __init__(self):
        self.sparse_attrs = ["time", "lon", "lat", "alt", "T", "RHw", "RHi"]
        self.time = SparseIndex(dtype="datetime64[ns]")
        self.lon = SparseIndex(dtype=np.float32)
        self.lat = SparseIndex(dtype=np.float32)
        self.alt = SparseIndex(dtype=np.float32)
        self.T = SparseIndex(dtype=np.float32)
        self.RHw = SparseIndex(dtype=np.float32)
        self.RHi = SparseIndex(dtype=np.float32)

        self.derived_attrs = ["coord"]
        self.coord = SparseIndex(dtype=np.float32)
        #self.instrument = SparseIndex()
        #self.instrument_type = SparseIndex()
    
    def copy_from(self, other : INPIndex,
                  loc : Optional[Union[List[int], int]] = None,
                  refactorise : bool = False
                  ):
        """Copy index values from another INPIndex,
        use loc to select a subset of point data
        refactorise to rebuild the index and keep only values present in the subset
        """

        if loc is not None and isinstance(loc, int):
            loc = [loc]
        for attr in self.sparse_attrs:
            other_sparse = getattr(other, attr)
            this_sparse = getattr(self, attr)
            
            if refactorise:
                new_index, new_uniques = pd.factorize(
                    other_sparse.getv(loc=loc), sort=False,
                    use_na_sentinel=False)
                this_sparse.index = new_index
                this_sparse.uniques = new_uniques
            else:
                if loc is not None:
                    this_sparse.index = list(np.asarray(other_sparse.index)[loc])
                else:
                    this_sparse.index = other_sparse.index.copy()
                this_sparse.uniques = other_sparse.uniques.copy()
    
        # Derived attrs
        self.set_coords()
    
    def store(self, attr : str, values : List[float]):
        """Store values for and build index"""

        existing = getattr(self, attr).uniques
        offset = len(existing)
        ####
        #TBD: check whether values are already in the existing uniques
        ####
        codes, these_uniques = pd.factorize(
            pd.Index(np.asarray(values)),
            sort=False,
            use_na_sentinel=False,
        )
        # Append indices
        getattr(self, f"{attr}").index.extend((codes + offset).tolist())
        setattr(getattr(self, attr), "uniques",
                np.append(existing, these_uniques.astype(existing.dtype)))


    def set_coords(self):
        """Sets unique coordinate indices
        based on current coordinate values
        """
        all_lons = self.getv("lon")
        all_lats = self.getv("lat")
        all_coords = np.array([(lon,lat)
                            for lon, lat in zip(
                                all_lons,
                                all_lats
                            )])
        uniques, coords_index = np.unique(all_coords, axis=0, return_inverse=True)

        self.coord.uniques = uniques
        self.coord.index = coords_index.tolist()
        self.coord.info["dims"] = "(point_idx, lon, lat)"
        self.coord.info["units"] = "(degrees east, degrees north)"

    @property
    def coord_lons(self):
        """Returns longitude values of the unique coordinates"""
        return self.coord.uniques[:,0]
    @property
    def coord_lats(self):
        """Returns latitude values of the unique coordinates"""
        return self.coord.uniques[:,1]


    def get_grid_xarray(self) -> xr.Dataset:
        """Returns xarray with the unique coordinate indexes"""

        if self.coord.is_empty():
            try:
                self.set_coords()
            except Exception as e:
                raise ValueError("Could not get grid info.") from e
        
        lon_metadata = {
            'long_name': 'Longitude',
            'units': 'degrees_east'
        }
        lat_metadata = {
            'long_name': 'Latitude',
            'units': 'degrees_north'
        }

        alt_metadata = {
            'long_name': 'Altitude',
            'units': 'm'
        }
        # Create xarray Dataset with coords and coord_index
        coords = {
            'lon': xr.DataArray(data=self.coord_lons, attrs=lon_metadata), # type: ignore
            'lat': xr.DataArray(data=self.coord_lats, attrs=lat_metadata), # type: ignore
            'time' : np.array(self.getv("time"))
        }
        data_vars = {
            'coord_index': ('time', self.coord.index),
            'alt': ('time', self.getv("alt"), alt_metadata),
            }
        ds = xr.Dataset(coords=coords,\
                        data_vars=data_vars)
        return ds
    
    def getv(self, attr : str, loc : Optional[Union[List[int], int]] = None):
        """Get unwrapped values for a given attribute and location(s)
        """
        if not hasattr(self, attr):
            raise ValueError(f"Attribute '{attr}' not found in index")
        sparse_entry = getattr(self, attr)

        return sparse_entry.getv(loc=loc)