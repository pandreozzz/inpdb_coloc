from __future__ import annotations

from typing import Optional, Union, List, Self

import numpy as np
import xarray as xr



class ModelHandler:
    """Handle model data for flexible colocations"""
    def __init__(self, coord_dim : str = "col", vert_dim : str = "lev"):
        self.data : Union[None, xr.Dataset] = None
        self.coord_dim = coord_dim
        self.vert_dim = vert_dim
    
    def get_subset(self,
                   coord_values : Optional[np.ndarray] = None,
                   atol=1.e-4,
                   ) -> Self:
        """Get a subset of the data based on coordinate values. This is used for colocation.
        By default a view of content of data is retuned
        coord_values if given must be a 2D array of shape (n,2),
        as stored in the uniques for the INP sparse structure 
        atol is the absolute tolerance for searching the coord_values
        """
        if self.data is None:
            return self
        subset = self.__class__(
            coord_dim=self.coord_dim,
            vert_dim=self.vert_dim
            )
        subset.coord_dim = self.coord_dim
        subset.vert_dim = self.vert_dim

        if coord_values is None:
            subset.data = self.data
        else:
            if self.data is None:
                raise ValueError("No data loaded. Cannot extract subset.")
            coord_model = np.asarray(
                [[lon, lat] for lon, lat in zip (
                    self.data.lon, self.data.lat
                    )])

            coord_model_indexes = []
            for ctgt in coord_values:
                matches = np.flatnonzero(np.isclose(coord_model, ctgt, atol=atol).all(axis=1))
                if len(matches) == 0:
                    raise ValueError(f"Failure getting subset. {ctgt} not found in model data.")
                elif len(matches) > 1:
                    raise ValueError(f"Failure getting subset. Multiple matches found for {ctgt} in model data.")
                coord_model_indexes.append(matches[0])
            subset.data = self.data.isel({self.coord_dim: coord_model_indexes})
            
        return subset
    
    def from_ncfile(self, clim_path : str) -> None:
        self.data = xr.open_dataset(clim_path)
    
class CamsClimHandler(ModelHandler):

    def get_data(self,
                 time : Union[np.datetime64,
                             List[np.datetime64],
                             xr.DataArray],
                 lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> Union[xr.DataArray, xr.Dataset]:
        """Returns the data, optionally selecting loc and time
        if time is provided as a DataArray, it can also have other coordinates along the time dimension, that will be considered align to time for the interpolation.
        """
        from src.utils.cams import interpolate_monthly_clim

        if self.data is None:
            raise ValueError("Data not loaded. Call from_cams_clim first.")

        if not isinstance(time, xr.DataArray):
            if not isinstance(time, list):
                time = [time]
            time = xr.DataArray(time, dims="time", coords={"time": time})


        this_data = self.data.isel({self.vert_dim : lev_idx}) if lev_idx is not None else self.data

        if loc is not None:
            this_data = this_data.sel({self.coord_dim: loc})

        if var_subset is not None:
            this_data = this_data[var_subset]

        return interpolate_monthly_clim(this_data, time)

class CamsOutputHandler(ModelHandler):
    def from_cams_output(self, output_path : Union[str, List[str]]):
        """Load data from CAMS output file(s). output_path can also have wildcards for multiple locations"""
        from src.utils.cams import RENAMEDIC
        from glob import glob
        if not isinstance(output_path, list):
            output_path = [output_path]
        all_files = []
        for path in output_path:
            all_files.extend(glob(path))
        all_files = list(set(all_files))

        if all_files == []:
            raise ValueError("No files found for the given output_path(s).")


        opends_kwargs = {"combine": "nested", "concat_dim": "time"}
        # Are they zarr archives?
        if all_files[0].endswith("zarr"):
            print("Opening archives as zarr")
            opends_kwargs["engine"] = "zarr"
        else:
            print("Opening files as netCDF")
            opends_kwargs["engine"] = "netcdf4"

        all_sfc_files = [f for f in all_files if "_sfc_" in f]
        all_pl_files = [f for f in all_files if "_pl_" in f]
        all_ml_files = [f for f in all_files if "_ml_" in f]

        this_data_list = []
        if all_sfc_files != []:
            print("Getting surface fields")
            this_data_list.append(xr.open_mfdataset(all_sfc_files, **opends_kwargs)) # type: ignore
        if all_pl_files != []:
            print("Getting pl fields")
            this_data_list.append(xr.open_mfdataset(all_pl_files, **opends_kwargs)) # type: ignore
        if all_ml_files != []:
            print("Getting ml fields")
            this_data_list.append(xr.open_mfdataset(all_ml_files, **opends_kwargs)) # type: ignore
        
        if this_data_list == []:
            raise ValueError("No files found for the given output_path(s).")

        this_data = xr.merge(this_data_list, compat="equals", join="outer")

        this_data = this_data.rename({k: v for k,v in RENAMEDIC.items() if k in this_data})

        self.data = this_data
    
    def get_data(self,
                 time : Union[np.datetime64, List[np.datetime64], xr.DataArray],
                 lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
        """Returns the data, optionally selecting loc and time. Nearest time is returned."""
        if self.data is None:
            raise ValueError("Data not loaded. Call from_cams_output first.")

        this_data = self.data.isel({self.vert_dim: lev_idx}) if lev_idx is not None else self.data
        if var_subset is not None:
            this_data = this_data[var_subset]

        sel_dict = {}
        if loc is not None:
            sel_dict[self.coord_dim] = loc
        if time is not None:
            nearest_times = this_data.time.sel(time=time, method="nearest")
            sel_dict["time"] = nearest_times
            # Co-location can be done by assigning the coord_dim coordinate inside time xarray
            if isinstance(time, xr.DataArray) and \
                self.coord_dim in time.coords and \
                    loc is None:
                sel_dict[self.coord_dim] = time[self.coord_dim]
        if sel_dict:
            this_data = this_data.sel(**sel_dict)

        return this_data