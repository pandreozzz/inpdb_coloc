from __future__ import annotations

from typing import Union, Optional, List, Tuple
import pandas as pd
import numpy as np
from src.physics.aerotools import get_nccn_over_mcon_from_specs
import xarray as xr

from .sparser import INPIndex
from .cores import DenseValue
from .model import CamsClimHandler, CamsOutputHandler
from src.physics.inp import INPParametrization
from src.physics.aerosol import AerosolSpec
    
class INPCollection:
    """Designed to contain sparse data.
    all _index arrays have the same length (one per entry from the csv), and each point to a value in the corresponding _s array.
    temperature and rel humidities are considered as sparse values (often the same due to instrumentation)
    INP_conc is non-indexed and a single value is stored per each entry 
    """

    def initialise(self):
        self.sparse = INPIndex()

        # These two must be consistent
        self.non_index_attrs = ['INP_conc']
        self.INP_conc = DenseValue()

        # Model data
        self.cams_clim = CamsClimHandler(vert_dim="lev")
        self.cams_free = CamsOutputHandler(vert_dim="plev")

        # Rejected entries
        self.rejected_entries : Optional[pd.DataFrame] = None
        
        self.coord_attrs = ['time', 'lon', 'lat', 'alt']
        self.value_attrs = ['T', 'RHw', 'RHi']+self.non_index_attrs #, 'instrument', 'instrument_type']
        self.all_iterable_attrs = self.coord_attrs + self.value_attrs #+\
                                        #['instrument', 'instrument_type']

    def get_subset(self, entry_indexes : Optional[List[int]] = None,
                   regenerate_coord_index : bool = True
                   ) -> INPCollection:
        """Obtain subset based on entry indexes (row numbers of the original csv)"""
        subset = INPCollection()

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

        return subset
    
    def copy(self) -> INPCollection:
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
                          ) -> INPCollection:
        """Extract subset region
        Returns the subset collection and the list of coordinate indexes
        """

        in_region = (self.lat() >= lat_south) & (self.lat() <= lat_north)
        if lon_west <= lon_east:
            in_region &= (self.lon() >= lon_west) & (self.lon() <= lon_east)
        else:
            # Handle wrap crossing
            in_region &= (self.lon() >= lon_west) | (self.lon() <= lon_east)

        entry_indexes = np.flatnonzero(in_region).tolist()
        return self.get_subset(entry_indexes,
                               regenerate_coord_index=regenerate_coord_index)
    
    def get_timerange_subset(self,
                             start_time : str, end_time : str,
                             regenerate_coord_index : bool = True
                             ) -> INPCollection:
        """Extract subset from timerange
        Returns the subset collection and the list of coordinate indexes
        """
        start, end = np.datetime64(start_time), np.datetime64(end_time)
        in_range = (self.time() >= start) & (self.time() <= end)
        entry_idx_sel = np.flatnonzero(in_range).tolist()

        return self.get_subset(entry_idx_sel,
                               regenerate_coord_index=regenerate_coord_index)
        
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
        self.INP_conc.values = np.append(self.INP_conc.values, df['IN [L-1]'].to_numpy(dtype=float))

    def load_cams_clim(self, clim_path : str):
        """Loads from file and aligns coordinates to the INP uniques"""
        self.cams_clim.from_ncfile(clim_path)
        self.cams_clim = self.cams_clim.get_subset(
            coord_values = self.sparse.coord.uniques
        )
    def load_cams_free(self, free_path : Union[str, List[str]]):
        """Loads from file and aligns coordinates to the INP uniques"""
        self.cams_free.from_cams_output(free_path)
        self.cams_free = self.cams_free.get_subset(
            coord_values = self.sparse.coord.uniques
        )
    
    def get_cams_clim(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                       loc : Union[None, int, List[int], xr.DataArray] = None,
                       var_subset : Union[None, str, List[str]] = None
                      ):
        """Get CAMS climatology interpolated to times."""

        # Default is exact co-location (space and time) with observations
        if time is None:
            time_extractor = xr.DataArray(
                data=self.time(),
                dims=["time"],
                coords={
                    "time": self.time(),
                    self.cams_clim.coord_dim : ("time", self.sparse.coord.index)
                    }
            )
        else:
            time_extractor = xr.DataArray(
                data=time,
                dims=["time"],
                coords={"time": time}
            )
        return self.cams_clim.get_data(time=time_extractor, loc=loc, var_subset=var_subset)
    
    def get_cams_free(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                      loc : Union[None, int, List[int], xr.DataArray] = None,
                      var_subset : Union[None, str, List[str]] = None
                      ):
        """Get CAMS free at requested times."""

        # Default is exact co-location (space and time) with observations
        if time is None:
            time_extractor = xr.DataArray(
                data=self.time(),
                dims=["time"],
                coords={
                    "time": self.time(),
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
        return self.cams_free.get_data(time=time_extractor, loc=loc, var_subset=var_subset)

    def get_cams_clim_inp(self, 
                          inp_params: dict[str, INPParametrization], 
                          aerosol_spec: AerosolSpec,
                          time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                          loc : Union[None, int, List[int], xr.DataArray] = None,
                          ) -> dict[str, Union[pd.Series, xr.DataArray]]:
        """Get CAMS climatology at the INP coordinates and times, and apply the parametrizations to get INP concentrations.
        Returns a dictionary with the INP concentrations per parametrization and the total INP concentration."""

        # Get cams aerosol mass mixing ratios
        cams_data = self.get_cams_free(time, loc)
        temperature_data = self.T()
        air_density_data = self.rho()

        inp_num_conc = {}
        for name, param in inp_params.items():
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
                          ) -> dict[str, Union[pd.Series, xr.DataArray]]:
        """Get CAMS free at the INP coordinates and times, and apply the parametrizations to get INP concentrations.
        Returns a dictionary with the INP concentrations per parametrization and the total INP concentration."""

        # Get cams aerosol mass mixing ratios
        cams_data = self.get_cams_clim(time, loc)
        temperature_data = self.T()
        air_density_data = self.rho()

        inp_num_conc = {}
        for name, param in inp_params.items():
            inp_num_conc[name] = param.compute_inp_concentration(temperature_data, air_density_data, cams_data, aerosol_spec)
            
        # Compute total INP concentration by summing over the different parametrizations
        total_inp_conc = sum(inp_num_conc.values())
        inp_num_conc["total"] = total_inp_conc

        # Merge dict to one dataset
        inp_num_conc_ds = xr.Dataset(
            data_vars = inp_num_conc
            )

        return inp_num_conc_ds

    def compact(self):
        print("Not implemented yet - should reprocess all indices and values to remove duplicates")
        pass

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

    def time(self, loc : Optional[Union[List[int], int]] = None):
        """Returns time per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("time", loc=loc)

    def lon(self, loc : Optional[Union[List[int], int]] = None):
        """Returns longitude per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("lon", loc=loc)

    def lat(self, loc : Optional[Union[List[int], int]] = None):
        """Returns latitude per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("lat", loc=loc)
    
    def alt(self, loc : Optional[Union[List[int], int]] = None):
        """Returns altitude per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("alt", loc=loc)
    
    def T(self, loc : Optional[Union[List[int], int]] = None):
        """Returns temperature per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("T", loc=loc)
    
    def RHw(self, loc : Optional[Union[List[int], int]] = None):
        """Returns relative humidity with respect to water per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("RHw", loc=loc)
    
    def RHi(self, loc : Optional[Union[List[int], int]] = None):
        """Returns relative humidity with respect to ice per each entry. entry indexes can be specified using loc"""
        return self.sparse.getv("RHi", loc=loc)

    def p(self, loc : Optional[Union[List[int], int]] = None):
        """Returns air pressure per each entry. entry indexes can be specified using loc. For now returns pressure at height from isothermal atmosphere. This should be revised to use actual pressure data e.g. from era5 or more accurate formula."""
        # barometric height formula for isothermal atmosphere
        from src.physics.constants import Constants
        p0 = 101325 # Pa
        constants = Constants()
        M = constants.M
        g = constants.g
        R = constants.R
        h = self.alt(loc=loc)
        T = self.T(loc=loc)
        return p0 * np.exp(-M * g * h / (R * T))
        

    def rho(self, loc : Optional[Union[List[int], int]] = None):
        """Returns air density per each entry. entry indexes can be specified using loc"""
        from src.physics.constants import Constants
        constants = Constants()
        # Ideal gas law to get air density from pressure and temperature
        air_density = self.p(loc=loc) / (constants.R_s * self.T(loc=loc)) 
        return air_density
    
    def __init__(self, inpdb_path : Optional[str] = None,
                 cams_clim_path : Optional[str] = None,
                 cams_free_path : Union[None, str, List[str]] = None,
                 n_lines : Optional[int] = None,
                 lonlat_approx : Optional[float] = None,
                 lon_to_degeast : bool = True,
                 t_approx_h : bool = True,
                 sortbytime : bool = False,
                 timerange: Optional[Tuple[str, str]] = None):
        self.initialise()
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
            
    
