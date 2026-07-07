from __future__ import annotations
import re
from typing import Dict, Optional, Union, List, Self

import numpy as np
import xarray as xr



class ModelHandler:
    """Handle model data for flexible colocations"""
    def __init__(self, coord_dim : str = "col", vert_dim : Optional[str] = "lev"):
        self.data : Union[None, xr.Dataset] = None
        self.coord_dim = coord_dim
        self.vert_dim = vert_dim

    def get_subset(self,
                   coord_values : Optional[np.ndarray] = None,
                   coords_to_assign : Dict[str, List] = {},
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
            model_lons = self.data["lon"].to_numpy()
            model_lats = self.data["lat"].to_numpy()
            coord_targets = np.asarray(coord_values, dtype=float)
            coord_scale = float(atol)

            coord_buckets: Dict[tuple[int, int], List[int]] = {}
            for model_idx, (lon, lat) in enumerate(zip(model_lons, model_lats)):
                key = tuple(np.rint(np.asarray((lon, lat)) / coord_scale).astype(int))
                coord_buckets.setdefault(key, []).append(model_idx)

            coord_model_indexes = []
            for ctgt in coord_targets:
                target_key = tuple(np.rint(ctgt / coord_scale).astype(int))
                candidate_indexes: List[int] = []
                for dlon in (-1, 0, 1):
                    for dlat in (-1, 0, 1):
                        candidate_indexes.extend(
                            coord_buckets.get((target_key[0] + dlon, target_key[1] + dlat), [])
                        )

                if candidate_indexes == []:
                    raise ValueError(f"Failure getting subset. {ctgt} not found in model data.")

                candidate_coords = np.column_stack((model_lons[candidate_indexes], model_lats[candidate_indexes]))
                candidate_deltas = np.abs(candidate_coords - ctgt)
                candidate_distances = np.max(candidate_deltas, axis=1)
                matches = np.flatnonzero(candidate_distances <= atol)
                if len(matches) == 0:
                    raise ValueError(f"Failure getting subset. {ctgt} not found in model data.")

                best_match = matches[np.argmin(candidate_distances[matches])]
                coord_model_indexes.append(candidate_indexes[int(best_match)])
            subset.data = self.data.isel({self.coord_dim: coord_model_indexes})

        for coord_name, coord_val in coords_to_assign.items():
            if len(coord_val) != subset.data.sizes[self.coord_dim]:
                raise ValueError(f"Length of coord_values for {coord_name} does not match the size of the subset along {self.coord_dim}.")
            subset.data = subset.data.assign_coords({coord_name: (self.coord_dim, coord_val)})

        return subset

    def from_ncfile(self, clim_path : str) -> None:
        self.data = xr.open_dataset(clim_path)

class MERRA2Handler(ModelHandler):
    def get_data(self,
                 time : Union[np.datetime64, List[np.datetime64], xr.DataArray],
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
        """Returns the data, optionally selecting loc and time. Nearest time is returned.
        if 2D data are handled, lev_idx must be None!
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call from_merra2_output first.")

        this_data = self.data
        if var_subset is not None:
            this_data = this_data[var_subset]

        sel_dict = {}
        if loc is not None:
            sel_dict[self.coord_dim] = loc

        if time is not None:
            this_data = this_data.sel(time=time, method="nearest").assign_coords({"time": time})

        return this_data.sel(**sel_dict) if sel_dict else this_data

class MacV2SPHandler(ModelHandler):

    def get_data(self,
                 time : Union[np.datetime64, List[np.datetime64], xr.DataArray],
                 lev_idx : Union[None, int, List[int], xr.DataArray] = None,
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
        """Returns the data, optionally selecting loc and time. Nearest time is returned.
        if 2D data are handled, lev_idx must be None!
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call from_macv2sp_output first.")

        this_data = self.data.isel({self.vert_dim: lev_idx}) if lev_idx is not None else self.data
        if var_subset is not None:
            this_data = this_data[var_subset]

        sel_dict = {}
        if loc is not None:
            sel_dict[self.coord_dim] = loc

        if time is not None:
            this_data = this_data.sel(time=time, method="nearest").assign_coords({"time": time})

        return this_data.sel(**sel_dict) if sel_dict else this_data


class CamsClimHandler(ModelHandler):

    def get_data(self,
                 time : Union[np.datetime64,
                             List[np.datetime64],
                             xr.DataArray],
                 lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
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

class ERA5DataHandler(ModelHandler):

    def from_era5_output(self, era5_path : str,
                         consolidated_zarr : bool = True) -> None:
        """Load data from ERA5 output file(s). era5_path can also have wildcards for multiple locations.
        Input data is a single file (zarr or netCDF) like era5_1980to2020_pl_monthly_aeronet.nc
        if the file is not found, it is created by looking for era5_1980_pl_monthly_aeronet.nc, ..., era5_2020_pl_monthly_aeronet.nc and merging them into a single file.
        """
        import os

        if not os.path.exists(era5_path):
            print(f"ERA5 data file {era5_path} not found. Looking for individual year files to merge.")
            import re
            this_match = re.match(r"era5_(\d{4})to(\d{4})_(\w+)_aeronet", os.path.basename(era5_path))
            if this_match is None or len(this_match.groups()) != 3:
                raise ValueError(f"Could not parse era5_path {era5_path}. Expected format: era5_YYYYtoYYYY_pl_<freq>_aeronet")

            st_year = int(this_match.group(1))
            end_year = int(this_match.group(2))
            aeronet_freq = this_match.group(3)

            from ..config import ERA5_DATADIR
            files_list_pl = []
            files_list_sp = []
            for year in range(st_year, end_year + 1):
                year_file_pl = os.path.join(ERA5_DATADIR, f"era5_{year}_pl_{aeronet_freq}_aeronet.nc")
                year_file_sp = os.path.join(ERA5_DATADIR, f"era5_{year}_sp_{aeronet_freq}_aeronet.nc")
                if not os.path.exists(year_file_pl):
                    raise ValueError(f"ERA5 data file for year {year} not found: {year_file_pl}")
                if not os.path.exists(year_file_sp):
                    raise ValueError(f"ERA5 data file for year {year} not found: {year_file_sp}")

                files_list_pl.append(year_file_pl)
                files_list_sp.append(year_file_sp)

            if files_list_pl == [] or files_list_sp == []:
                raise ValueError("No files (pl and sp) found for the given era5_path(s).")

            opends_kwargs = {"combine": "nested", "concat_dim": "time",
                             "engine" : "netcdf4", "chunks": {"time": 25}}

            xr.merge(
                [xr.open_mfdataset(files_list_pl, **opends_kwargs),
                 xr.open_mfdataset(files_list_sp, **opends_kwargs)]
            ).to_zarr( # type: ignore
                era5_path, mode="w",
                consolidated=consolidated_zarr,
                )
            print(f"Created merged ERA5 data file: {era5_path}")

        opends_kwargs = {}
        print(f"Loading ERA5 data from {era5_path}")
        if era5_path.endswith("zarr"):
            opends_kwargs["engine"] = "zarr"
            opends_kwargs["consolidated"] = consolidated_zarr
        else:
            opends_kwargs["engine"] = "netcdf4"
        self.data = xr.open_dataset(era5_path, **opends_kwargs) # type: ignore

        if "_monthly_" in os.path.basename(era5_path):
            # Setting day to 15 of each month for monthly data
            if "time" in self.data.coords:
                self.data["time"] = self.data["time"].dt.floor("D") + np.timedelta64(14, "D")


    def get_data(self,
                 time : Union[np.datetime64, List[np.datetime64], xr.DataArray],
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
        """Returns the data, optionally selecting loc and time. Nearest time is returned."""
        if self.data is None:
            raise ValueError("Data not loaded. Call from_era5_output first.")

        this_data = self.data

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