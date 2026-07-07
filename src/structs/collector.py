from __future__ import annotations

from typing import Dict, Union, Optional, List, Tuple, Self, Literal
import os


import pandas as pd
import numpy as np
import xarray as xr

from tqdm import tqdm

from ..config import AeronetFreq, AeronetLevel
from .sparser import SparseIndexedCollection, INPIndexedCollection, AeronetIndexedCollection
from .cores import DenseValue
from .model import ModelHandler, CamsClimHandler, CamsOutputHandler, \
      ERA5DataHandler, MacV2SPHandler, MERRA2Handler
from ..physics.inp import INPParametrization
from ..physics.aerosol import AerosolSpec

from ..config import get_cams_clim_path, get_cams_clim_sites_path, get_cams_clim_aeronet_path,  \
    get_cams_free_sites_path, get_cams_free_aeronet_path, get_era5_aeronet_path


class ObsCollection:
    """Designed to contain sparse data. (Mixing many campaigns)
    all _index arrays have the same length (one per entry from the csv), and each point to a value in the corresponding _s array.
    temperature and rel humidities are considered as sparse values (often the same due to instrumentation)
    INP_conc is non-indexed and a single value is stored per each entry
    """
    dense_values = []
    extra_sparse_values = []
    gettable_sparse_vars = ["time", "lon", "lat", "alt"] # geolocation



    griddes_file : str = ""
    cams_clim_file : str = get_cams_clim_path()

    cams_clim_points_file : str = ""
    cams_free_points_file : str = ""
    era5_points_file : str = ""
    era5_points_geopot_file : str = ""

    def __init__(self, sparse_struct : Optional[SparseIndexedCollection] = None) -> None:
        if sparse_struct is not None:
            self.sparse = sparse_struct

    @classmethod
    def add_sparse_getter(cls, name):
        def sparse_getter(self, loc : Optional[Union[List[int], int]] = None):
            return self.sparse.getv(name, loc=loc)
        setattr(cls, name, sparse_getter)

    def __init_subclass__(cls):
        super().__init_subclass__()

        sparse_vars_all = []
        for c in reversed(cls.__mro__):
            sparse_vars_all.extend(getattr(c, "gettable_sparse_vars", []))

        for v in set(sparse_vars_all):
            cls.add_sparse_getter(v)

    def initialise(self, non_index_attrs : List[str] = [], extra_sparse_values : List[str] = []):

        # These two must be consistent
        self.non_index_attrs = non_index_attrs
        for non_index_attr in self.non_index_attrs:
            if not hasattr(self, non_index_attr):
                setattr(self, non_index_attr, DenseValue())

        # Model data
        self.cams_clim = CamsClimHandler(vert_dim="lev")
        self.cams_free = CamsOutputHandler(vert_dim="plev")
        self.era5 = ERA5DataHandler(vert_dim="plev")

        # Rejected entries
        self.rejected_entries : Optional[pd.DataFrame] = None

        self.coord_attrs = ['time', 'lon', 'lat', 'alt']
        self.value_attrs = self.extra_sparse_values+self.non_index_attrs #, 'instrument', 'instrument_type']
        self.all_iterable_attrs = self.coord_attrs + self.value_attrs #+\
                                        #['instrument', 'instrument_type']

    def get_subset(self, entry_indexes : Optional[List[int]] = None,
                   regenerate_coord_index : bool = True
                   ) -> Self:
        """Obtain subset based on entry indexes (row numbers of the original csv)"""
        subset = self.__class__()
        if not hasattr(subset, "sparse"):
            subset.sparse = self.sparse.__class__()

        subset.sparse.copy_from(self.sparse, loc=entry_indexes,
                                refactorise=True)
        for attr in self.non_index_attrs:
            getattr(subset, attr).copy_from(getattr(self, attr), loc=entry_indexes)

        if not regenerate_coord_index:
            for attr in ["index", "uniques"]:
                setattr(
                    subset.sparse.coord,
                    attr,
                    getattr(self.sparse.coord, attr).copy()
                    )
            coord_indexes = None

        # The fields are stored along the coord points
        subset.cams_clim = self.cams_clim.get_subset(coord_values = subset.sparse.coord.uniques)
        subset.cams_free = self.cams_free.get_subset(coord_values = subset.sparse.coord.uniques)
        subset.era5 = self.era5.get_subset(coord_values = subset.sparse.coord.uniques)

        return subset

    def copy(self) -> Self:
        """Obtain a copy of the collection"""
        return self.get_subset(entry_indexes=None,
                               regenerate_coord_index=False)

    def get_coord_subset(self, coord_indexes : List[int],
                         regenerate_coord_index : bool = True
                         ) -> None: # INPCollection:
        """Obtain subset based on unique coordinate indexes"""
        # Not used and might be wrong
        pass
        # entry_indexes = list(np.flatnonzero(np.isin(
        #     np.asarray(self.sparse.coord.index),
        #     np.asarray(coord_indexes))).astype(int))
        # return self.get_subset(entry_indexes,
        #                        regenerate_coord_index=regenerate_coord_index)

    def get_region_subset(self,
                          lon_west : float, lon_east : float,
                          lat_south : float, lat_north : float,
                          regenerate_coord_index : bool = True
                          ) -> Self:
        """Extract subset region
        Returns the subset collection and the list of coordinate indexes
        """

        lats = self.sparse.getv("lat")
        lons = self.sparse.getv("lon")

        in_region = (lats >= lat_south) & (lats <= lat_north)
        if lon_west <= lon_east:
            in_region &= (lons >= lon_west) & (lons <= lon_east)
        else:
            # Handle wrap crossing
            in_region &= (lons >= lon_west) | (lons <= lon_east)

        entry_indexes = np.flatnonzero(in_region).tolist()
        return self.get_subset(entry_indexes,
                               regenerate_coord_index=regenerate_coord_index)

    def get_timerange_subset(self,
                             start_time : str, end_time : str,
                             regenerate_coord_index : bool = True
                             ) -> Self:
        """Extract subset from timerange
        Returns the subset collection and the list of coordinate indexes
        """
        start, end = np.datetime64(start_time), np.datetime64(end_time)
        times = self.sparse.getv("time")
        in_range = (times >= start) & (times <= end)
        entry_idx_sel = np.flatnonzero(in_range).tolist()

        return self.get_subset(entry_idx_sel,
                               regenerate_coord_index=regenerate_coord_index)

    def load_cams_clim(self, clim_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_clim.from_ncfile(clim_path)
        self.cams_clim = self.cams_clim.get_subset(
            coord_values = self.sparse.coord.uniques
        )
    def load_cams_free(self, free_path : Union[str, List[str]]):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_free.from_cams_output(free_path)
        self.cams_free = self.cams_free.get_subset(
            coord_values = self.sparse.coord.uniques
        )
    def load_era5(self, era5_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.era5.from_era5_output(era5_path)
        self.era5 = self.era5.get_subset(
            coord_values = self.sparse.coord.uniques
        )

    def load_era5_geopot(self, era5_geopot_path: str):
        """Load surface geopotential aligned to the already loaded ERA5 points."""
        era5_geopot = ERA5DataHandler(vert_dim=self.era5.vert_dim)
        era5_geopot.from_ncfile(era5_geopot_path)
        era5_geopot = era5_geopot.get_subset(
            coord_values=self.sparse.coord.uniques
        )

        assert self.era5.data is not None
        assert era5_geopot.data is not None
        self.era5.data["z_sfc"] = era5_geopot.data["z"].squeeze(drop=True)

    def get_cams_clim(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                       lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                       loc : Union[None, int, List[int], xr.DataArray] = None,
                       var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS climatology interpolated to times."""

        # Default is exact co-location (space and time) with observations
        obs_times = self.sparse.getv("time")
        if time is None:
            time_extractor = xr.DataArray(
                data=obs_times,
                dims=["time"],
                coords={
                    "time": obs_times,
                    self.cams_clim.coord_dim : ("time", self.sparse.coord.index)
                    }
            )
        else:
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        return self.cams_clim.get_data(time=time_extractor, lev_idx=lev_idx,
                                       loc=loc, var_subset=var_subset)

    def get_cams_free(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                      lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                      loc : Union[None, int, List[int], xr.DataArray] = None,
                      var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS free at requested times."""

        # Default is exact co-location (space and time) with observations
        obs_times = self.sparse.getv("time")
        if time is None:
            time_extractor = xr.DataArray(
                data=obs_times,
                dims=["time"],
                coords={
                    "time": obs_times,
                    self.cams_free.coord_dim : ("time", self.sparse.coord.index)
                    }
            )
        elif not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        else:
            time_extractor = time
        return self.cams_free.get_data(time=time_extractor, lev_idx=lev_idx,
                                       loc=loc, var_subset=var_subset)

    def get_grid_xarray(self) -> xr.Dataset:
        return self.sparse.get_grid_xarray()


    def to_xarray(self, lead_dim : str = 'time',
                  loc : Optional[Union[int, List[int]]] = None
                  ) -> xr.Dataset:
        """Convert to xarray Dataset
        """

        if loc is not None and isinstance(loc, int):
            loc = [loc]

        if lead_dim not in self.coord_attrs:
            raise ValueError(f"Lead dimension '{lead_dim}' must be one of {self.coord_attrs}")

        # Create coordinate arrays
        coords = {attr: (lead_dim, self.sparse.getv(attr, loc=loc))
                  for attr in self.coord_attrs}
        # Create data variables
        data_vars = {attr: (lead_dim, self.sparse.getv(attr, loc=loc))
                     for attr in self.value_attrs
                     if attr not in self.non_index_attrs}
        for attr in self.non_index_attrs:
            data_vars[attr] = (lead_dim, getattr(self, attr).values[loc] if loc is not None else getattr(self, attr).values)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        if "time" in ds:
            ds["time"] = ds["time"].astype("datetime64[ns]")
        return ds


class INPCollection(ObsCollection):
    """Subclass for INP data"""

    dense_values = ["INP_conc"]
    sparse_values = ["T", "RHw", "RHi"]
    gettable_sparse_vars = sparse_values

    cams_clim_points_file = get_cams_clim_sites_path()
    cams_free_points_file = get_cams_free_sites_path()

    def __init__(self, inpdb_path : Optional[str] = None,
                 cams_clim_path : Optional[str] = None,
                 cams_free_path : Union[None, str, List[str]] = None,
                 n_lines : Optional[int] = None,
                 lonlat_approx : Optional[float] = None,
                 lon_to_degeast : bool = True,
                 t_approx_h : bool = True,
                 sortbytime : bool = False,
                 timerange: Optional[Tuple[str, str]] = None):

        super().__init__(sparse_struct=INPIndexedCollection())

        for dense_val in self.dense_values:
            setattr(self, dense_val, DenseValue())

        super().initialise(non_index_attrs=self.dense_values,
                           extra_sparse_values=self.sparse_values)

        if inpdb_path is not None:
            print(f"Loading INP collection from {inpdb_path}")
            self.load_inp_collection(inpdb_path, n_lines,
                                     lonlat_approx=lonlat_approx,
                                     lon_to_degeast=lon_to_degeast,
                                     t_approx_h=t_approx_h, sortbytime=sortbytime,
                                     timerange=timerange)
        if cams_clim_path is not None:
            print(f"Loading CAMS climatology from {cams_clim_path}")
            self.load_cams_clim(cams_clim_path)
        if cams_free_path is not None:
            print(f"Loading CAMS free output from {cams_free_path}")
            self.load_cams_free(cams_free_path)

    def load_inp_collection(self, file_path: str,
                            n_lines: Optional[int] = None,
                            t_approx_h: bool = True,
                            lonlat_approx : Optional[float] = None,
                            lon_to_degeast : bool = True,
                            sortbytime : bool = False,
                            timerange: Optional[Tuple[str, str]] = None,
                            ):
        """Digest INP data from CSV file
        """
        df = pd.read_csv(file_path, nrows=n_lines)
        assert file_path.endswith(".csv"), "Input file must be a CSV file"
        self.griddes_file = file_path.replace(".csv", "_griddes.txt")

        # Build datetime column vectorially
        tstr = df['Time'].str.split(':').apply(
            lambda parts: ':'.join(p.zfill(2) for p in parts) if isinstance(parts, list) else None
        )
        # Year, Month, Day required
        for col in ['Year', 'Month', 'Day']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop all invalid entries
            df = df[df[col].notna()]

        time_str_col = (
            df['Year'].astype(int).astype(str).str.zfill(4) + '-' +
            df['Month'].astype(int).astype(str).str.zfill(2) + '-' +
            df['Day'].astype(int).astype(str).str.zfill(2) + 'T' +
            tstr.fillna('00:00:00')
        )
        times = pd.to_datetime(time_str_col, errors='coerce')
        if t_approx_h:
            times = times.dt.floor('h')

        # Drop year, month, day columns and assign time
        df = df.drop(columns=['Year', 'Month', 'Day', 'Time'])
        df['time'] = times
        del times

        if sortbytime:
            df = df.sort_values('time').reset_index(drop=True)

        # Drop entries with invalid time strings
        df_nas = df[df.time.isna()]
        df = df[df.time.notna()]
        if len(df_nas) > 0:
            print(f"Warning: Dropped {len(df_nas.time)} rows with invalid time strings "+\
                  " - storing into rejected_entries.")
            self.rejected_entries = df_nas

        # Drop out of timerange
        if timerange is not None:
            start, end = pd.to_datetime(timerange[0]), pd.to_datetime(timerange[1])

            in_range = (df.time >= start) & (df.time <= end)
            df = df[in_range]

        #self.instruments.extend(df['Instrument'].tolist())
        #self.instrument_types.extend(df['Instrument_Type'].tolist())

        # Build unique-value lists and per-row indexes via factorize
        col_map = {
            'time': df['time'].values,
            'lon':  df['Lon'].to_numpy(dtype=float),
            'lat':  df['Lat'].to_numpy(dtype=float),
            'alt':  df['Alt [m]'].to_numpy(dtype=float),
            'T':    df['T [K]'].to_numpy(dtype=float),
            'RHw':  df['RHw [%]'].to_numpy(dtype=float),
            'RHi':  df['RHi [%]'].to_numpy(dtype=float),
            #'instrument': df['Instrument'].to_numpy(dtype=str),
            #'instrument_type': df['Instrument_Type'].to_numpy(dtype=str)
        }
        if lonlat_approx is not None:
            for attr in ["lon", "lat"]:
                values = col_map[attr]
                col_map[attr] = np.round(values / lonlat_approx) * lonlat_approx

        if lon_to_degeast:
            col_map["lon"] = np.mod(col_map["lon"], 360)

        # Populate the sparse 1D index
        for attr, values in col_map.items():
            self.sparse.store(attr, values)

        # Populate coords
        self.sparse.set_coords()

        # The dense values
        #self.INP_conc.values = np.append(self.INP_conc.values, df['IN [L-1]'].to_numpy(dtype=float))
        for dense_attr in self.dense_values:
            dense_values = df[dense_attr].to_numpy(dtype=float)
            getattr(self, dense_attr).values = np.append(getattr(self, dense_attr).values, dense_values)

    def get_cams_clim_inp(self,
                          inp_params: dict[str, INPParametrization],
                          aerosol_spec: AerosolSpec,
                          time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                          loc : Union[None, int, List[int], xr.DataArray] = None,
                          T_source: str = "inpdb" # or "era5" to be implemented,
                          ) -> xr.Dataset:
        """Get CAMS climatology at the INP coordinates and times, and apply the parametrizations to get INP concentrations.
        Returns an xarray Dataset with one variable per parametrization plus a `total` variable."""

        # Get cams aerosol mass mixing ratios
        cams_data = self.get_cams_clim(time, loc)
        temperature_data = None
        if T_source == "inpdb":
            temperature_data = self.sparse.getv("T")
        elif T_source == "era5":
            # Implement ERA5 temperature extraction here. This would allow using locations and times not co-located with the INP observations
            pass
        air_density_data = self.rho() # should use model pressure and temperature in the future
        inp_num_conc = {}
        for name, param in inp_params.items():
            assert temperature_data is not None, "Temperature data required for compute_inp_concentration"
            inp_num_conc[name] = param.compute_inp_concentration(temperature_data, air_density_data, cams_data, aerosol_spec)

        # Compute total INP concentration by summing over the different parametrizations
        total_inp_conc = sum(inp_num_conc.values())
        inp_num_conc["total"] = total_inp_conc

        # Merge dict to one dataset
        inp_num_conc_ds = xr.Dataset(
            data_vars = inp_num_conc
            )

        return inp_num_conc_ds

    def get_cams_free_inp(self,
                          inp_params: dict[str, INPParametrization],
                          aerosol_spec: AerosolSpec,
                          time : Union[None, np.datetime64,
                                       List[np.datetime64],
                                       np.ndarray,
                                       xr.DataArray] = None,
                          loc : Union[None, int, List[int], xr.DataArray] = None,
                          T_source: str = "inpdb" # or "era5" to be implemented,
                          ) -> xr.Dataset:
        """Get CAMS free-running output at the INP coordinates and times, and apply the parametrizations to get INP concentrations.
        Returns an xarray Dataset with one variable per parametrization plus a `total` variable."""

        # Get cams aerosol mass mixing ratios
        temperature_data = None
        cams_data = self.get_cams_free(time, loc)
        if T_source == "inpdb":
            temperature_data = self.sparse.getv("T")
        elif T_source == "era5":
            # Implement ERA5 temperature extraction here. This would allow using locations and times not co-located with the INP observations
            pass
        air_density_data = self.rho() # should use model pressure and temperature in the future
        inp_num_conc = {}
        for name, param in inp_params.items():
            assert temperature_data is not None, "Temperature data required for compute_inp_concentration"
            inp_num_conc[name] = param.compute_inp_concentration(temperature_data, air_density_data, cams_data, aerosol_spec)

        # Compute total INP concentration by summing over the different parametrizations
        total_inp_conc = sum(inp_num_conc.values())
        inp_num_conc["total"] = total_inp_conc

        # Merge dict to one dataset
        inp_num_conc_ds = xr.Dataset(
            data_vars = inp_num_conc
            )

        return inp_num_conc_ds

    def p(self, loc : Optional[Union[List[int], int]] = None):
        """Returns air pressure per each entry. entry indexes can be specified using loc. For now returns pressure at height from isothermal atmosphere. This should be revised to use actual pressure data e.g. from era5 or more accurate formula."""
        # barometric height formula for isothermal atmosphere
        from src.physics.constants import Constants
        p0 = 101325 # Pa
        constants = Constants()
        M = constants.M
        g = constants.g
        R = constants.R
        h = self.sparse.getv("alt", loc=loc)
        T = self.sparse.getv("T", loc=loc)
        return p0 * np.exp(-M * g * h / (R * T))

    def rho(self, loc : Optional[Union[List[int], int]] = None):
        """Returns air density per each entry. entry indexes can be specified using loc"""
        from src.physics.constants import Constants
        constants = Constants()
        # Ideal gas law to get air density from pressure and temperature
        air_density = self.p(loc=loc) / (constants.R_s * self.sparse.getv("T", loc=loc))
        return air_density


class ObsStation:
    """Contains one Aeronet station"""
    def __init__(self, station_name : str,
                 lon : float, lat : float, alt : float):
        self.station_name = station_name
        self.lon = lon
        self.lat = lat
        self.alt = alt


class StationsCollection:
    """Contain station data. Every station has fixed location, and content is in a separate object."""

    griddes_file : str = ""
    cams_clim_file : str = get_cams_clim_path()

    cams_clim_points_file : str = ""
    cams_free_points_file : str = ""
    era5_points_file : str = ""
    era5_points_geopot_file : str = ""

    # Model data
    cams_clim = CamsClimHandler(vert_dim="lev")
    cams_free = CamsOutputHandler(vert_dim="plev")
    era5 = ERA5DataHandler(vert_dim="plev")

    # Other clims
    macv2sp = MacV2SPHandler(vert_dim=None)
    merra2 = MERRA2Handler(vert_dim=None)

    # Rejected entries
    rejected_entries : Optional[pd.DataFrame] = None

    # Each station is identified by name
    stations : Dict[str, ObsStation] = {}

    # No need to touch this
    coord_atol : float = 1.e-3

    def __init__(self):
        self.stations = {}
        self.cams_clim = CamsClimHandler(vert_dim="lev")
        self.cams_free = CamsOutputHandler(vert_dim="plev")
        self.era5 = ERA5DataHandler(vert_dim="plev")

    def _swap_model_dim_to_station_name(self, model_handler: ModelHandler) -> None:
        """Promote the station_name coordinate to the active horizontal dimension."""
        if model_handler.data is None or "station_name" not in model_handler.data.coords:
            return


        dim_1d = model_handler.data["station_name"].dims[0]
        if dim_1d != "station_name":
            model_handler.data = model_handler.data.swap_dims({dim_1d: "station_name"})
        model_handler.coord_dim = "station_name"

    def get_subset(self, stations : Optional[List[str]] = None) -> Self:
        """Obtain subset based on station names"""
        subset = self.__class__()

        if stations is None:
            subset.stations = self.stations.copy()
        else:
            subset.stations = {name: self.stations[name]
                               for name in stations
                               if name in self.stations}
            # Check which stations could not be found
            missing_stations = [name for name in stations if name not in self.stations]
            if missing_stations:
                print(f"Warning: The following stations were not found in the collection: {missing_stations}")
        stations_coords = np.asarray(subset.stations_coords)
        get_subset_kwargs = {"coord_values": stations_coords,
                             "coords_to_assign": {"station_name": subset.station_names},
                             "atol": self.coord_atol}
        subset.cams_clim = self.cams_clim.get_subset(**get_subset_kwargs)
        subset.cams_free = self.cams_free.get_subset(**get_subset_kwargs)
        subset.era5 = self.era5.get_subset(**get_subset_kwargs)
        subset.macv2sp = self.macv2sp.get_subset(**get_subset_kwargs)
        subset.merra2 = self.merra2.get_subset(**get_subset_kwargs)

        # Now swap_dims to station_name
        for model_dset_name in ["cams_clim", "cams_free", "era5"]:
            model_dset = getattr(subset, model_dset_name)
            subset._swap_model_dim_to_station_name(model_dset)
        return subset

    def copy(self) -> Self:
        """Obtain a copy of the collection"""
        return self.get_subset(stations=None)

    @property
    def stations_lats(self) -> List[float]:
        """Return the latitudes of all stations in the collection"""
        return [station.lat for station in self.stations.values()]
    @property
    def stations_lons(self) -> List[float]:
        """Return the longitudes of all stations in the collection"""
        return [station.lon for station in self.stations.values()]

    @property
    def stations_coords(self) -> List[Tuple[float, float]]:
        """Return the coordinates of all stations in the collection
        list of pairs (lon, lat)
        """
        return [(lon, lat) for lon, lat in zip(self.stations_lons, self.stations_lats)]

    @property
    def station_alts(self) -> List[float]:
        """Return the altitudes of all stations in the collection"""
        return [station.alt for station in self.stations.values()]

    @property
    def station_names(self) -> List[str]:
        """Return the names of all stations in the collection"""
        return list(self.stations.keys())

    def get_region_subset(self,
                          lon_west : float, lon_east : float,
                          lat_south : float, lat_north : float
                          ) -> Self:
        """Extract subset region
        Returns the subset collection and the list of coordinate indexes
        """
        stations_in_region = [name for name, station in self.stations.items()
                              if lon_west <= station.lon <= lon_east and lat_south <= station.lat <= lat_north]
        return self.get_subset(stations=stations_in_region)

    def load_merra2(self, merra2_path: str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.merra2.from_ncfile(merra2_path)
        self.merra2 = self.merra2.get_subset(
            coord_values=np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol
        )
        self._swap_model_dim_to_station_name(self.merra2)

    def load_macv2sp(self, macv2sp_path: str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.macv2sp.from_ncfile(macv2sp_path)
        self.macv2sp = self.macv2sp.get_subset(
            coord_values=np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol
        )
        self._swap_model_dim_to_station_name(self.macv2sp)

    def load_cams_clim(self, clim_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_clim.from_ncfile(clim_path)

        self.cams_clim = self.cams_clim.get_subset(
            coord_values = np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol
        )
        self._swap_model_dim_to_station_name(self.cams_clim)

    def load_cams_free(self, free_path : Union[str, List[str]]):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_free.from_cams_output(free_path)
        self.cams_free = self.cams_free.get_subset(
            coord_values = np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol
        )
        self._swap_model_dim_to_station_name(self.cams_free)

    def load_era5(self, era5_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.era5.from_era5_output(era5_path)
        self.era5 = self.era5.get_subset(
            coord_values = np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol
        )
        self._swap_model_dim_to_station_name(self.era5)

    def load_era5_geopot(self, era5_geopot_path: str):
        """Load surface geopotential aligned to the station-expanded ERA5 points."""
        era5_geopot = ERA5DataHandler(vert_dim=self.era5.vert_dim)
        era5_geopot.from_ncfile(era5_geopot_path)
        era5_geopot = era5_geopot.get_subset(
            coord_values=np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol,
        )
        self._swap_model_dim_to_station_name(era5_geopot)

        assert self.era5.data is not None
        assert era5_geopot.data is not None
        self.era5.data["z_sfc"] = era5_geopot.data["z"].squeeze(drop=True)

    def get_merra2(self,
                   time : Union[np.datetime64,
                                List[np.datetime64],
                                np.ndarray,
                                xr.DataArray],
                   loc : Union[None, int, List[int], xr.DataArray] = None,
                   var_subset : Union[None, str, List[str]] = None
                   ) -> xr.Dataset:
        """Get MERRA2 data at requested times."""

        assert self.get_merra2 is not None, "MERRA2 data handler is not loaded yet!"

        if isinstance(time, np.datetime64):
            time = [time]

        # Default is exact co-location (space and time) with observations
        if not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        else:
            time_extractor = time
        return self.merra2.get_data(time=time_extractor, loc=loc, var_subset=var_subset)


    def get_macv2sp(self,
                    time : Union[np.datetime64,
                                 List[np.datetime64],
                                 np.ndarray,
                                 xr.DataArray],
                    loc : Union[None, int, List[int], xr.DataArray] = None,
                    var_subset : Union[None, str, List[str]] = None
                    ) -> xr.Dataset:
        """Get MACv2-SP data at requested times."""

        assert self.get_macv2sp is not None, "MACv2-SP data handler is not loaded yet!"

        if isinstance(time, np.datetime64):
            time = [time]

        # Default is exact co-location (space and time) with observations
        if not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        else:
            time_extractor = time
        return self.macv2sp.get_data(time=time_extractor, loc=loc, var_subset=var_subset)

    def get_cams_clim(self,
                      time : Union[np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray],
                       lev_idx : Union[None, int, List[int], xr.DataArray] = None,
                       loc : Union[None, int, List[int], xr.DataArray] = None,
                       var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS climatology interpolated to times."""

        if isinstance(time, np.datetime64):
            time = [time]

        # Default is exact co-location (space and time) with observations
        if not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        else:
            time_extractor = time

        return self.cams_clim.get_data(time=time_extractor, lev_idx=lev_idx,
                                       loc=loc, var_subset=var_subset)

    def get_cams_free(self,
                      time : Union[np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray],
                      lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                      loc : Union[None, int, List[int], xr.DataArray] = None,
                      var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS free at requested times."""

        # Default is exact co-location (space and time) with observations
        if not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        else:
            time_extractor = time
        return self.cams_free.get_data(time=time_extractor, lev_idx=lev_idx,
                                       loc=loc, var_subset=var_subset)

    @property
    def coord_lons(self) -> np.ndarray:
        """Return the longitudes of all stations in the collection"""
        return np.array([station.lon for station in self.stations.values()])

    @property
    def coord_lats(self) -> np.ndarray:
        """Return the latitudes of all stations in the collection"""
        return np.array([station.lat for station in self.stations.values()])

    @property
    def coord_alts(self) -> np.ndarray:
        """Return the altitudes of all stations in the collection"""
        return np.array([station.alt for station in self.stations.values()])

    def get_grid_xarray(self) -> xr.Dataset:
        """Returns xarray with the unique coordinate indexes"""

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

        station_coords = np.asarray(self.stations_coords)
        # Approximate the coordinates to atol before getting uniques
        station_coords = np.round(station_coords / self.coord_atol) * self.coord_atol
        station_coords = np.unique(station_coords, axis=0)
        # Create xarray Dataset with coords and coord_index
        coords = {
            'lon': xr.DataArray(data=station_coords[:, 0], attrs=lon_metadata), # type: ignore
            'lat': xr.DataArray(data=station_coords[:, 1], attrs=lat_metadata), # type: ignore
        }
        ds = xr.Dataset(coords=coords,\
                        data_vars=None)
        # Ensure coordinate uniqueness

        return ds

    def compute_p_station(self, rho_0 : float = 1.225, hscale_m : float = 8400.0) -> xr.DataArray:
        """Compute station surface pressure from ERA5 geopotential and scale height
        rho_0 : sea level air density
        hscale_m : scale height
        """
        from src.physics.constants import CONST_G, CONST_Rd

        if self.era5.data is None:
            raise ValueError("ERA5 data not loaded yet!")
        for var in ["sp", "z_sfc"]:
            if var not in self.era5.data.data_vars:
                raise ValueError(f"ERA5 data must contain '{var}' for the station surface pressure")

        sp = self.era5.data["sp"]
        alt0 = self.era5.data["z_sfc"] / CONST_G
        alt_st = xr.DataArray(
            data=self.coord_alts,
            dims=["station_name"],
            coords={"station_name": self.station_names}
        )

        return sp + rho_0 * CONST_G * hscale_m *\
             (np.exp(-alt_st / hscale_m) - np.exp(-alt0 / hscale_m))


class AeronetCollection(StationsCollection):
    """Collection of Aeronet stations"""
    aod_wl_defaults = [340, 380, 440, 500, 550, 675, 870, 1020]
    ae_wl_defaults = [(440, 870), (380, 500), (440, 675), (500, 870), (340, 440), (440, 675)]

    # cams_clim_points_file : str = get_cams_clim_aeronet_path()
    # cams_free_points_file : str = get_cams_free_aeronet_path()
    # macv2sp_points_file : str = get_macv2sp_aeronet_path()

    def __init__(self, aeronet_dir_path : Optional[str] = None,
                 cams_clim_path : Optional[str] = None,
                 cams_free_path : Union[None, str, List[str]] = None,
                 macv2sp_path : Optional[str] = None,
                 merra2_path : Optional[str] = None,
                 timerange: Optional[Tuple[str, str]] = None,
                 aod_wl : List[int] = aod_wl_defaults,
                 ae_wl : List[Tuple[int, int]] = ae_wl_defaults
                ):
        super().__init__()

        self.aod_maps = {
            f"AOD{wl:d}": f"AOD_{wl:d}nm"
            for wl in aod_wl
        }
        self.ae_maps = {
            f"AE_{wl[0]:d}-{wl[1]:d}": f"{wl[0]:d}-{wl[1]:d}_Angstrom_Exponent"
            for wl in ae_wl
        }

        self.aod_wl = aod_wl
        self.ae_wl = ae_wl

        from ..config import CONFIGDICT
        aeronet_freq = CONFIGDICT.get("aeronet_freq", None)
        aeronet_lev = CONFIGDICT.get("aeronet_lev", None)

        if aeronet_freq is None or aeronet_lev is None:
            print("No aeronet_freq or aeronet_lev in CONFIGDICT, using defaults: monthly and lev20")
            aeronet_freq = "monthly"
            aeronet_lev = "lev20"

        self.era5_points_file = get_era5_aeronet_path(aeronet_freq=aeronet_freq)

        if aeronet_dir_path is not None:
            print(f"Loading Aeronet collection from {aeronet_dir_path}")
            self.load_aeronet_collection(aeronet_dir_path,
                                         header_lines=6,
                                         aeronet_freq=aeronet_freq,
                                         aeronet_lev=aeronet_lev,
                                         timerange=timerange)

        if cams_clim_path != "" and cams_clim_path is not None:
            print(f"Loading CAMS climatology from {cams_clim_path}")
            # try:
            self.load_cams_clim(cams_clim_path)
            # except Exception as exc:
            #     print(f"Warning: Failed to load CAMS climatology from {cams_clim_path}: {exc}")

        if cams_free_path != "" and cams_free_path is not None:
            print(f"Loading CAMS free output from {cams_free_path}")
            # try:
            self.load_cams_free(cams_free_path)
            # except Exception as exc:
            #     print(f"Warning: Failed to load CAMS free output from {cams_free_path}: {exc}")

        if macv2sp_path != "" and macv2sp_path is not None:
            print(f"Loading MACv2-SP from {macv2sp_path}")
            # try:
            self.load_macv2sp(macv2sp_path)
            # except Exception as exc:
            #     print(f"Warning: Failed to load MACv2-SP from {macv2sp_path}: {exc}")

        if merra2_path != "" and merra2_path is not None:
            print(f"Loading MERRA2 from {merra2_path}")
            # try:
            self.load_merra2(merra2_path)
            # except Exception as exc:
            #     print(f"Warning: Failed to load MERRA2 from {merra2_path}: {exc}")

        if aeronet_dir_path is not None and self.era5_points_file != "" and self.era5_points_file is not None:
            print(f"Loading ERA5 output from {self.era5_points_file}")
            # try:
            self.load_era5(self.era5_points_file)
            # except Exception as exc:
            #     print(f"Warning: Failed to load ERA5 output from {self.era5_points_file}: {exc}")

            self.era5_points_geopot_file = get_era5_aeronet_path(aeronet_freq=aeronet_freq,
                                                            geopot=True)
            print(f"Loading ERA5 geopotential output from {self.era5_points_geopot_file}")
            # try:
            self.load_era5_geopot(self.era5_points_geopot_file)
            # except Exception as exc:
            #     print(f"Warning: Failed to load ERA5 geopotential from {self.era5_points_geopot_file}: {exc}")

    def load_aeronet_collection(self, aeronet_dir_path: str,
                                header_lines: int = 6,
                                aeronet_freq : Literal["monthly", "daily"] = "monthly",
                                aeronet_lev : Literal["lev20", "lev15", "lev10"] = "lev20",
                                timerange: Optional[Tuple[str, str]] = None):
        """Load Aeronet collection from directory containing CSV files"""

        from glob import glob


        # Check for existence of synthesis file
        synth_file_name_like = os.path.join(aeronet_dir_path, "synth_files", f"synthfile_{aeronet_freq}_{aeronet_lev}_*.parquet")
        synth_files = glob(synth_file_name_like)
        synth_files.sort()

        if synth_files == []:
            # Get most recent
            print(f"No synthesis file found at {synth_file_name_like}, generating from all data. Will take a while")
            all_aeronet_files = glob(os.path.join(aeronet_dir_path, f"*.{aeronet_lev}"))
            all_aeronet_files.sort()

            all_aeronet_dfs = []
            for file_path in tqdm(all_aeronet_files):
                all_aeronet_dfs.append(self.open_aeronet_csv(file_path,
                                                             header_lines=header_lines,
                                                             aeronet_freq=aeronet_freq,
                                                             aeronet_lev=aeronet_lev,
                                                             timerange=timerange))

            df_merged = pd.concat(all_aeronet_dfs, ignore_index=True)
            date = pd.Timestamp.now().strftime("%Y%m%d")
            synth_file = synth_file_name_like.replace("*", date)
            df_merged.to_parquet(os.path.join(aeronet_dir_path, synth_file))
            del df_merged

        else:
            synth_file = synth_files[-1]
            print(f"Found synthesis file {synth_file}")

        df_merged = pd.read_parquet(synth_file)
        if aeronet_freq == "monthly":
            # Set date to the 15 of the month
            df_merged["time"] = df_merged["time"].apply(lambda x: x.replace(day=15))
        self.df_merged = df_merged

        self.griddes_file = synth_file.replace(".parquet", "_griddes.txt")

        # Drop entries with invalid time strings
        df_nas = df_merged[df_merged.time.isna()]
        df_merged = df_merged[df_merged.time.notna()]
        if len(df_nas) > 0:
            print(f"Warning: Dropped {len(df_nas.time)} rows with invalid time strings "+\
                    " - storing into rejected_entries.")
            self.rejected_entries = df_nas

        self.df_merged = df_merged

        # Build the stations catalogue from one grouped pass instead of
        # rescanning the full dataframe once per station.
        station_meta = df_merged.groupby("station_name", sort=False).agg(
            lon=("Longitude(degrees)", "first"),
            lat=("Latitude(degrees)", "first"),
            alt=("Elevation(meters)", "first"),
            lon_nunique=("Longitude(degrees)", "nunique"),
            lat_nunique=("Latitude(degrees)", "nunique"),
            alt_nunique=("Elevation(meters)", "nunique"),
        )

        for station_name_key, row in station_meta.iterrows():
            station_name = str(station_name_key)
            if row["lon_nunique"] != 1 or row["lat_nunique"] != 1:
                print(f"Warning: Station {station_name} has multiple unique lon/lat values.")
            if row["alt_nunique"] != 1:
                print(f"Warning: Station {station_name} has multiple unique alt values.")

            self.stations[station_name] = ObsStation(
                station_name=station_name,
                lon=float(row["lon"]),
                lat=float(row["lat"]),
                alt=float(row["alt"])
            )



    def open_aeronet_csv(self, file_path: str,
                         header_lines : Optional[int] = 6,
                         aeronet_freq : Literal["monthly", "daily"] = "monthly",
                         aeronet_lev : Literal["lev20", "lev15", "lev10"] = "lev20",
                         timerange: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Open Aeronet CSV file and return a DataFrame"""
        import os
        import re

        # Extract ini_date end_date
        match = re.match(r"(\d{8})_(\d{8})_(.+)\." + aeronet_lev, os.path.basename(file_path))
        assert match is not None, f"Filename {file_path} does not match expected pattern for Aeronet files."
        # ini_date = match.group(1)
        # end_date = match.group(2)
        station_name = match.group(3)

        df = pd.read_csv(file_path, header=header_lines, encoding="latin1")


        # Decode Month written in "YYYY-MON" format
        if aeronet_freq == "monthly":
            df["time"] = pd.to_datetime(df["Month"], format="%Y-%b")
            # Drop Month and make time the first column
            df = df.drop(columns=["Month"])
            df = df[["time"] + [col for col in df.columns if col != "time"]]

        # Get all AOD columns:
        all_aod_cols = [col for col in df.columns if re.match(r"AOD_\d+nm", col)]
        all_ae_cols = [col for col in df.columns if re.match(r"\d+-\d+_Angstrom_Exponent", col)]
        all_met_cols = [v for v in ["Precipitable_Water(cm)"] if v in df.columns]
        all_num_days_cols = [col for col in df.columns if re.match(r"NUM_DAYS\[.+\]", col)]
        all_num_pnts_cols = [col for col in df.columns if re.match(r"NUM_POINTS\[.+\]", col)]

        all_columns = ["time", "Longitude(degrees)", "Latitude(degrees)", "Elevation(meters)"] +\
                all_aod_cols + all_ae_cols + all_met_cols + all_num_days_cols + all_num_pnts_cols
        #print(f"Loading columns: {all_columns}")
        df = df[all_columns]

        df["station_name"] = station_name

        # Longitude to degrees east
        df["Longitude(degrees)"] = np.mod(df["Longitude(degrees)"], 360)

        # Drop out of timerange
        if timerange is not None:
            start, end = pd.to_datetime(timerange[0]), pd.to_datetime(timerange[1])

            in_range = (df.time >= start) & (df.time <= end)
            df = df[in_range]

        # replace -999 with NaN
        df.replace(-999, np.nan, inplace=True)
        return df

    def get_subset(self, stations: Optional[List[str]] = None) -> Self:
        """Obtain subset, also filtering df_merged to the selected stations."""
        subset = super().get_subset(stations=stations)
        if hasattr(self, "df_merged") and self.df_merged is not None:
            subset.df_merged = self.df_merged[
                self.df_merged["station_name"].isin(subset.station_names)
            ].copy()
        # Copy scalar attributes
        for attr in ["aod_wl", "ae_wl", "aod_maps", "ae_maps"]:
            if hasattr(self, attr):
                setattr(subset, attr, getattr(self, attr))
        return subset

    def get_stations_subset(self, station_names: List[str]) -> Self:
        """Obtain subset based on station names"""
        return self.get_subset(stations=station_names)

    def get_obs_aod(self,
                    include_precipitable_water: bool = True,
                    angstrom_fill : bool = True
                    ) -> xr.Dataset:
        """Return observed AERONET AOD (and optionally AE / precipitable water)
        as an xarray Dataset with dimensions ``(time, station_name)``.

        Parameters
        ----------
        wavelengths_nm
            Subset of wavelengths to include.  Defaults to all available.
        include_ae
            Include Angstrom Exponent columns when present.
        include_precipitable_water
            Include precipitable water column when present.
        """
        import re

        if not hasattr(self, "df_merged") or self.df_merged is None:
            raise ValueError("No AERONET data loaded — call load_aeronet_collection first.")

        df = self.df_merged.copy()

        # Restrict to stations currently in the collection (respects get_stations_subset)
        df = df[df["station_name"].isin(self.station_names)]

        # Identify column names
        aod_cols = [c for c in df.columns if re.match(r"AOD_\d+nm", c)]
        #wl_tags = {f"AOD_{wl}nm" for wl in self.aod_wl}
        #aod_wl = [wl for wl in self.aod_wl if f"AOD_{wl}nm" in aod_cols]
        aewl_tags = {f"{wl[0]}-{wl[1]}_Angstrom_Exponent" for wl in self.ae_wl}
        ae_cols = [c for c in df.columns if c in aewl_tags]

        extra_cols: List[str] = []
        if include_precipitable_water and "Precipitable_water(cm)" in df.columns:
            extra_cols.append("Precipitable_water(cm)")

        value_cols = aod_cols + ae_cols + extra_cols
        pivot_cols = ["time", "station_name"] + value_cols
        df = df[pivot_cols].copy()
        df["time"] = pd.to_datetime(df["time"])

        # Pivot to (time × station_name) grid
        ds = (
            df.set_index(["time", "station_name"])[value_cols]
            .to_xarray()
        )

        # Attach lon / lat / alt as coordinates
        lons = {name: st.lon for name, st in self.stations.items()}
        lats = {name: st.lat for name, st in self.stations.items()}
        alts = {name: st.alt for name, st in self.stations.items()}
        station_names_in_ds = ds["station_name"].values
        ds = ds.assign_coords(
            lon=("station_name", [lons[n] for n in station_names_in_ds]),
            lat=("station_name", [lats[n] for n in station_names_in_ds]),
            alt=("station_name", [alts[n] for n in station_names_in_ds]),
        )

        # Build AOD DataArray — keep as standalone to avoid Dataset re-alignment
        # eating the newly added wavelengths when assigning back via ds["AOD"] = ...
        aod_da = xr.concat(
            [ds[aod_var].expand_dims(
                wavelength_nm=[int(re.search(r"AOD_(\d+)nm", aod_var).group(1))] #type: ignore
            ) for aod_var in aod_cols],
            dim="wavelength_nm"
        )

        # All wavelength labels stay as integers for consistent label lookup.
        target_wl = np.asarray(self.aod_wl, dtype=int)
        obs_wl = aod_da["wavelength_nm"].values.astype(int)
        # support_wl = observed ∪ target; keeps bracketing wavelengths available for Angstrom fill
        support_wl = np.unique(np.concatenate([obs_wl, target_wl]))
        aod_da = aod_da.assign_coords(wavelength_nm=obs_wl)
        aod_da = aod_da.reindex(wavelength_nm=support_wl, fill_value=np.nan).sortby("wavelength_nm") # type: ignore

        # Fill missing target wavelengths using Angstrom interpolation (support wavelengths available as brackets)
        if angstrom_fill:
            from ..physics.optics import angstrom_fill_nans
            for wl_tgt in target_wl:
                angstrom_fill_nans(aod_da, wl_tgt=wl_tgt, wl_dim="wavelength_nm", max_wl_dist=200, verbose=True)

        aod_da = aod_da.sel(wavelength_nm=target_wl)  # drop support-only wavelengths
        ds["AOD"] = aod_da

        ds["AE"] = xr.concat(
            [ds[ae_var].expand_dims(
                wavelength_ae_nm=["-".join(re.search(r"(\d+)-(\d+)_Angstrom_Exponent", ae_var).groups())] #type: ignore
            ) for ae_var in ae_cols],
            dim="wavelength_ae_nm"
        )
        ds = ds.drop_vars(aod_cols + ae_cols)  # drop original columns

        return ds

    _AODSources = Literal["cams_clim", "cams_clim_natural", "cams_free", "macv2sp", "merra2"]
    def calculate_aod(self,
                      times : Union[None, np.ndarray, xr.DataArray] = None,
                      source : _AODSources = "cams_clim",
                      chunk : Literal["year", "month"] = "year",
                      wavelengths_nm : Optional[Tuple[int, ...]] = None,
                      per_species : bool = False,
                      aero_opt_ver : str = "48r1_4dclim",
                      optics_lut_path : Optional[str] = None,
                      clip_z_to_station : bool = True,
                      recompute : bool = False,
                      n_workers : int = 1,
                      show_progress : bool = True):
        """Compute offline AOD from the aerosol fields and ERA5 relative humidity.

        For each target time the nearest-in-time ERA5 relative humidity profile
        is selected, the optics LUT is interpolated to it (Fortran interface),
        and the column AOD is integrated.  Results are cached on the instance to
        avoid recomputing on repeated calls; pass ``recompute=True`` to refresh.

        Parameters
        ----------
        times
            Target times.  Defaults to the ERA5 time axis.
        source
            Aerosol source: ``"cams_clim"`` (default) or ``"cams_free"``.
        chunk
            ``"year"`` (default) computes per calendar year; ``"month"`` splits
            into per-month computations (lower memory).
        wavelengths_nm
            Wavelengths (nm) at which to evaluate AOD.
        per_species
            Also return per-species ``AOD_<spec>`` / ``Abs_<spec>``.
        clip_z_to_station
            Clip the integration to the station surface pressure (estimated from
            ERA5 ``sp``/``z_sfc`` and the station altitude) so only the column
            above the station contributes.
        recompute
            Ignore any cached result and recompute.
        """
        from ..physics.optics import AODCalculator

        if not hasattr(self, "_aod_cache"):
            self._aod_cache = {}

        if wavelengths_nm is None:
            wavelengths_nm = tuple(self.aod_wl)

        if times is None:
            time_key = None
            assert self.era5.data is not None
            times = self.era5.data["time"]
        else:
            tvals = np.asarray(getattr(times, "values", times), dtype="datetime64[ns]")
            time_key = (str(tvals.min()), str(tvals.max()), int(tvals.size))
        cache_key = (source, chunk, tuple(wavelengths_nm), bool(per_species),
                     aero_opt_ver, time_key, bool(clip_z_to_station))

        if not recompute and cache_key in self._aod_cache:
            return self._aod_cache[cache_key]

        if source == "cams_clim":
            handler = self.get_cams_clim(time=times, lev_idx=None)  # all levels
            vert_dim = self.cams_clim.vert_dim
        elif source == "cams_clim_natural":
            natural_aerospecs = [f"Sea_Salt_bin{i}" for i in [1,2,3]] +\
                                [f"Mineral_Dust_bin{i}" for i in [1,2,3]]
            handler = self.get_cams_clim(time=times, lev_idx=None)[natural_aerospecs + ["pressure"]]
            vert_dim = self.cams_clim.vert_dim
        elif source == "cams_free":
            handler = self.get_cams_free(time=times, lev_idx=None)  # all levels
            vert_dim = self.cams_free.vert_dim
        elif source == "macv2sp":
            handler = None
            result = self.calculate_aod(times=times, source="cams_clim_natural", chunk=chunk,
                                             wavelengths_nm=wavelengths_nm,
                                             per_species=False,
                                             aero_opt_ver=aero_opt_ver,
                                             optics_lut_path=optics_lut_path,
                                             clip_z_to_station=clip_z_to_station,
                                             recompute=recompute,
                                             n_workers=n_workers,
                                             show_progress=show_progress).rename({"AOD": "AOD_natural"})
            result["AOD_anthropogenic"] = self.get_macv2sp(time=times)["aod_2D"]
            result["AOD"] = result["AOD_natural"] + result["AOD_anthropogenic"]

            self._aod_cache[cache_key] = result

            return self._aod_cache[cache_key]
        elif source == "merra2":
            handler = None
            result = self.get_merra2(time=times)[["AODANA"]].rename(
                {"AODANA": "AOD"}
                ).expand_dims(wavelength=[550])
            self._aod_cache[cache_key] = result

            return self._aod_cache[cache_key]
        else:
            raise ValueError(f"Invalid source '{source}', must be one of {self._AODSources}")

        if self.era5.data is None or "r" not in self.era5.data:
            raise ValueError("ERA5 relative humidity is not loaded for this collection")

        if times is None:
            times = self.era5.data["time"]

        calc = AODCalculator(optics_lut_path=optics_lut_path,
                             aero_opt_ver=aero_opt_ver,
                             wavelengths_nm=wavelengths_nm)
        surface_pressure = None
        if clip_z_to_station:
            surface_pressure = self.compute_p_station()

        assert vert_dim is not None, "A vertical dimension is needed to compute AOD!"
        result = calc.compute_aod(handler, self.era5.data,
                                  target_times=times, chunk=chunk,
                                  per_species=per_species,
                                  lev_dim=vert_dim,
                                  surface_pressure=surface_pressure,
                                  n_workers=n_workers,
                                  show_progress=show_progress)
        # Add accessory coordinates from era5
        era5_coords = self.era5.data.coords
        for coord in era5_coords:
            coord_dims = era5_coords[coord].dims
            if len(coord_dims) == 1 and coord_dims[0] in result.dims and coord not in result.coords:
                result = result.assign_coords({coord: era5_coords[coord]})

        from src.physics.volcaero import get_glossac_aod
        try:
            print("Adding volcanic aerosol AOD from GloSSAC dataset")
            glossac_aod = get_glossac_aod().interp(lat=result.lat, method="linear").sel(wavelength_nm=result.wavelength, method="nearest")
            if result.time.min() < glossac_aod.time.min() - np.timedelta64(365, "D") or \
                result.time.max() > glossac_aod.time.max() + np.timedelta64(365, "D"):
                print("Warning: AOD volcanic data is more than 1 year out of bounds of the requested times.")
            result["AOD_volc_glossac"] = glossac_aod.sel(time=result.time, method="nearest")
        except Exception as exc:
            print(f"Warning: Failed to load volcanic AOD data: {exc}")
            result["AOD_volc_glossac"] = np.nan

        if source.startswith("cams_clim"):
            result = result.rename({"AOD_total": "AOD_tropo"})
            result["AOD"] = result["AOD_tropo"] + result["AOD_volc_glossac"]


        ae_values = []
        for ae_wl in self.ae_wl:
            ae_coord = f"{ae_wl[0]}-{ae_wl[1]}"
            ae_values.append((np.log(result.sel(wavelength=ae_wl[0])["AOD"] / result.sel(wavelength=ae_wl[1])["AOD"]) / np.log(ae_wl[1] / ae_wl[0])).expand_dims(wavelength_ae=[ae_coord]))

        result["AE"] = xr.concat(ae_values, dim="wavelength_ae")

        self._aod_cache[cache_key] = result

        return self._aod_cache[cache_key]
