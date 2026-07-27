from __future__ import annotations

from typing import Dict, Union, Optional, List, Tuple, Self, Literal, cast
import os


import pandas as pd
import numpy as np
import xarray as xr

from tqdm import tqdm

from ..config import AeronetFreq, AeronetLevel
from .sparser import SparseIndexedCollection, INPIndexedCollection, AeronetIndexedCollection
from .cores import DenseValue
from .model import ModelHandler, CamsClimHandler, CamsOutputHandler, \
      ERA5DataHandler, MacV2SPHandler, MacV2NatHandler, MERRA2Handler
from ..physics.inp import INPParametrization
from ..physics.aerosol import AerosolSpec

from ..config import get_cams_clim_path, get_cams_clim_sites_path, get_cams_clim_aeronet_path,  \
    get_cams_free_sites_path, get_cams_free_aeronet_path, get_era5_aeronet_path, get_era5_inpdb_path


class ObsCollection:
    """Designed to contain sparse data. (Mixing many campaigns)
    all _index arrays have the same length (one per entry from the csv), and each point to a value in the corresponding _s array.
    temperature and rel humidities are considered as sparse values (often the same due to instrumentation)
    INP_conc is non-indexed and a single value is stored per each entry
    """
    obs_index_dim: str = "obs_id"  # xarray dimension name for the per-observation axis
    _own_dense_values: List[str] = [obs_index_dim]
    _own_dense_dtypes: Dict = {obs_index_dim: np.int64}
    dense_values: List[str] = [obs_index_dim]       # accumulated by __init_subclass__ for subclasses
    dense_dtypes: Dict = {obs_index_dim: np.int64}  # accumulated by __init_subclass__ for subclasses
    extra_sparse_values = []
    gettable_sparse_vars = ["time", "lon", "lat", "alt"] # geolocation

    coord_atol : float = 1.e-4  # tolerance for matching unique (lon, lat) to model grid points

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

        # Accumulate dense_values and dense_dtypes from MRO so each class only
        # declares its own additions via _own_dense_values / _own_dense_dtypes.
        all_dense: List[str] = []
        all_dtypes: Dict = {}
        for c in reversed(cls.__mro__):
            all_dense.extend(c.__dict__.get("_own_dense_values", []))
            all_dtypes.update(c.__dict__.get("_own_dense_dtypes", {}))
        cls.dense_values = list(dict.fromkeys(all_dense))
        cls.dense_dtypes = dict(all_dtypes)

    def initialise(self, non_index_attrs : List[str] = [], extra_sparse_values : List[str] = []):

        # These two must be consistent
        self.non_index_attrs = non_index_attrs
        for non_index_attr in self.non_index_attrs:
            if not hasattr(self, non_index_attr):
                dtype = self.dense_dtypes.get(non_index_attr, np.float64)
                setattr(self, non_index_attr, DenseValue(dtype=dtype))

        # Model data
        self.cams_clim = CamsClimHandler(vert_dim="lev")
        self.cams_free = CamsOutputHandler(vert_dim="plev")
        self.era5 = ERA5DataHandler(vert_dim="plev")

        # Rejected entries
        self.rejected_entries : Optional[pd.DataFrame] = None

        self.coord_attrs = ['time', 'lon', 'lat', 'alt']
        self.value_attrs = list(set(extra_sparse_values+self.extra_sparse_values+self.non_index_attrs)) #, 'instrument', 'instrument_type']
        self.all_iterable_attrs = self.coord_attrs + self.value_attrs #+\
                                        #['instrument', 'instrument_type']

    def _make_time_extractor(self, coord_dim: str) -> xr.DataArray:
        """Build a DataArray used to index model data along the observation axis.
        ``obs_id`` is the primary dimension; ``time`` is a non-dimension coordinate
        so it never conflicts with model time axes after nearest-time selection.
        """
        obs_times = self.sparse.getv("time")
        obs_dim = self.obs_index_dim
        obs_id = getattr(self, obs_dim).values  # type: ignore[attr-defined]
        return xr.DataArray(
            data=obs_times,
            dims=[obs_dim],
            coords={
                obs_dim: obs_id,
                coord_dim: (obs_dim, self.sparse.coord.index)
            }
        )

    def _make_time_extractor_explicit(self, coord_dim: str, time) -> xr.DataArray:
        """Build a time extractor for an explicitly provided (non-None) time argument.
        Default: returns a plain ``time``-dimension DataArray.
        Override in subclasses to change the indexing strategy (e.g. ``obs_id × time``).
        """
        if isinstance(time, xr.DataArray):
            return time
        return xr.DataArray(data=time, dims=["time"], coords={"time": time})

    def _with_obs_times(self, ds: xr.Dataset, time_is_none: bool) -> xr.Dataset:
        """When co-locating (``time=None``), attach observation timestamps as a
        non-dimension ``'time'`` coordinate on the ``obs_id`` axis so callers can
        identify which observation each row corresponds to.
        When ``time`` was explicitly provided the dataset already carries the
        requested time axis; nothing is added.
        """
        if not time_is_none:
            return ds
        obs_dim = self.obs_index_dim
        return ds.assign_coords({"time": (obs_dim, self.sparse.getv("time"))})

    def get_subset(self, entry_indexes : Optional[List[int]] = None) -> Self:
        """Obtain subset based on entry indexes (row numbers of the original csv)"""
        subset = self.__class__()
        if not hasattr(subset, "sparse"):
            subset.sparse = self.sparse.__class__()

        subset.sparse.copy_from(self.sparse, loc=entry_indexes,
                                refactorise=True)
        for attr in self.non_index_attrs:
            getattr(subset, attr).copy_from(getattr(self, attr), loc=entry_indexes)

        # The fields are stored along the coord points
        subset.cams_clim = self.cams_clim.get_subset(coord_values=subset.sparse.coord.uniques, atol=self.coord_atol)
        subset.cams_free = self.cams_free.get_subset(coord_values=subset.sparse.coord.uniques, atol=self.coord_atol)
        subset.era5 = self.era5.get_subset(coord_values=subset.sparse.coord.uniques, atol=self.coord_atol)

        return subset

    def copy(self) -> Self:
        """Obtain a copy of the collection"""
        return self.get_subset(entry_indexes=None)

    def get_region_subset(self,
                          lon_west : float, lon_east : float,
                          lat_south : float, lat_north : float
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
        return self.get_subset(entry_indexes)

    def get_timerange_subset(self,
                             start_time : str, end_time : str
                             ) -> Self:
        """Extract subset from timerange
        Returns the subset collection and the list of coordinate indexes
        """
        start, end = np.datetime64(start_time), np.datetime64(end_time)
        times = self.sparse.getv("time")
        in_range = (times >= start) & (times <= end)
        entry_idx_sel = np.flatnonzero(in_range).tolist()

        return self.get_subset(entry_idx_sel)

    def load_cams_clim(self, clim_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_clim.from_ncfile(clim_path)
        self.cams_clim = self.cams_clim.get_subset(
            coord_values=self.sparse.coord.uniques, atol=self.coord_atol
        )
    def load_cams_free(self, free_path : Union[str, List[str]]):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.cams_free.from_cams_output(free_path)

        self.cams_free = self.cams_free.get_subset(
            coord_values=self.sparse.coord.uniques, atol=self.coord_atol
        )
    def load_era5(self, era5_path : str):
        """Loads from file and aligns coordinates to the Obs uniques"""
        self.era5.from_era5_output(era5_path)
        self.era5 = self.era5.get_subset(
            coord_values=self.sparse.coord.uniques, atol=self.coord_atol
        )

    def load_era5_geopot(self, era5_geopot_path: str):
        """Load surface geopotential aligned to the already loaded ERA5 points."""
        era5_geopot = ERA5DataHandler(vert_dim=self.era5.vert_dim)
        era5_geopot.from_ncfile(era5_geopot_path)
        era5_geopot = era5_geopot.get_subset(
            coord_values=self.sparse.coord.uniques, atol=self.coord_atol
        )

        if self.era5.data is not None and era5_geopot.data is not None:
            self.era5.data["z_sfc"] = era5_geopot.data["z"].squeeze(drop=True)

    def get_cams_clim(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                      vert_interp : bool = True,
                      lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                      loc : Union[None, int, List[int], xr.DataArray] = None,
                      var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS climatology interpolated to times.
        vert_interp: whether the data are vertically interpolated to the observation altitude
        lev_idx: specify the vertical level (only if vert_interp is False)
        """
        from src.physics.vinterp import interp

        # Default is exact co-location (space and time) with observations
        if time is None:
            time_extractor = self._make_time_extractor(self.cams_clim.coord_dim)
        else:
            time_extractor = self._make_time_extractor_explicit(self.cams_clim.coord_dim, time)


        if vert_interp:
            from src.physics.vinterp import interp_xarray
            lev_dim = self.cams_clim.vert_dim
            assert lev_dim is not None

            # Step 1: pressure only (all levels, 1 variable)
            p_cams = self.cams_clim.get_data(
                time=time_extractor, lev_idx=None, loc=loc, var_subset=["pressure"]
            )["pressure"]  # (lev, time), ascending pressure

            p_obs = self.compute_p_obs()  # (time,)
            tgtlevs, weights = interp_xarray(
                p_src=p_cams,
                p_tgt=p_obs.expand_dims({lev_dim: [0]}),
                intp_dim_name=lev_dim,
            )
            tgtlevs = tgtlevs.squeeze(lev_dim)  # (time,)
            weights = weights.squeeze(lev_dim)  # (time,)

            # Step 2: load only the unique bracket levels
            # tgtlevs from Fortran are 1-based; subtract 1 for 0-based isel.
            n_lev = p_cams.sizes[lev_dim]
            lower = np.clip(tgtlevs.values.astype(int) - 1, 0, n_lev - 2)
            upper = lower + 1
            unique_levs = np.unique(np.concatenate([lower, upper]))
            cams_slice = self.cams_clim.get_data(
                time=time_extractor, lev_idx=unique_levs.tolist(),
                loc=loc, var_subset=var_subset
            )

            # Step 3: locate lower/upper within the loaded slice and interpolate
            # tgtlevs may be 1-D (obs_id,) for co-location or 2-D (obs_id, time)
            # for explicit time — derive dims/coords directly from tgtlevs.
            tgt_dims = list(tgtlevs.dims)
            tgt_coords = {d: tgtlevs[d] for d in tgt_dims}
            lower_pos = xr.DataArray(
                np.searchsorted(unique_levs, lower),
                dims=tgt_dims, coords=tgt_coords
            )
            upper_pos = xr.DataArray(
                np.searchsorted(unique_levs, upper),
                dims=tgt_dims, coords=tgt_coords
            )
            return self._with_obs_times(
                (1.0 - weights) * cams_slice.isel({lev_dim: lower_pos}) +
                 weights         * cams_slice.isel({lev_dim: upper_pos}),
                time is None)

        return self._with_obs_times(
            self.cams_clim.get_data(
                time=time_extractor, lev_idx=lev_idx,
                loc=loc, var_subset=var_subset),
            time is None)



    def get_cams_free(self,
                      time : Union[None, np.datetime64,
                                   List[np.datetime64],
                                   np.ndarray,
                                   xr.DataArray] = None,
                      vert_interp : bool = True,
                      lev_idx : Union[None, int, List[int], xr.DataArray] = -1,
                      loc : Union[None, int, List[int], xr.DataArray] = None,
                      var_subset : Union[None, str, List[str]] = None
                      ) -> xr.Dataset:
        """Get CAMS free at requested times."""

        # Default is exact co-location (space and time) with observations
        if time is None:
            time_extractor = self._make_time_extractor(self.cams_free.coord_dim)
        else:
            time_extractor = self._make_time_extractor_explicit(self.cams_free.coord_dim, time)

        if vert_interp:
            from src.physics.vinterp import interp_xarray
            lev_dim = self.cams_free.vert_dim
            assert lev_dim == "plev", "Only pressure-level coordinate supported for vertical interpolation!"

            # plev is a fixed 1D coordinate — access directly, no get_data needed
            assert self.cams_free.data is not None
            p_cams = self.cams_free.data[lev_dim]  # (plev,), ascending pressure

            p_obs = self.compute_p_obs()  # (time,)
            tgtlevs, weights = interp_xarray(
                p_src=p_cams,
                p_tgt=p_obs.expand_dims({lev_dim: [0]}),
                intp_dim_name=lev_dim,
            )
            tgtlevs = tgtlevs.squeeze(lev_dim)  # (time,)
            weights = weights.squeeze(lev_dim)  # (time,)

            # Load only the unique bracket levels
            n_lev = p_cams.sizes[lev_dim]
            lower = np.clip(tgtlevs.values.astype(int) - 1, 0, n_lev - 2)
            upper = lower + 1
            unique_levs = np.unique(np.concatenate([lower, upper]))
            cams_slice = self.cams_free.get_data(
                time=time_extractor, lev_idx=unique_levs.tolist(),
                loc=loc, var_subset=var_subset
            )

            obs_dim = self.obs_index_dim
            tgt_dims = list(tgtlevs.dims)
            tgt_coords = {d: tgtlevs[d] for d in tgt_dims}
            lower_pos = xr.DataArray(
                np.searchsorted(unique_levs, lower),
                dims=tgt_dims, coords=tgt_coords
            )
            upper_pos = xr.DataArray(
                np.searchsorted(unique_levs, upper),
                dims=tgt_dims, coords=tgt_coords
            )
            return self._with_obs_times(
                (1.0 - weights) * cams_slice.isel({lev_dim: lower_pos}) +
                 weights         * cams_slice.isel({lev_dim: upper_pos}),
                time is None)


        return self._with_obs_times(
            self.cams_free.get_data(time=time_extractor, lev_idx=lev_idx,
                                    loc=loc, var_subset=var_subset),
            time is None)

    def get_era5(self,
                 time : Union[None, np.datetime64,
                              List[np.datetime64],
                              np.ndarray,
                              xr.DataArray] = None,
                 lev_idx : Union[None, int, List[int], xr.DataArray] = None,
                 loc : Union[None, int, List[int], xr.DataArray] = None,
                 var_subset : Union[None, str, List[str]] = None
                 ) -> xr.Dataset:
        """Get ERA5 data at requested times."""

        # Default is exact co-location (space and time) with observations
        if time is None:
            time_extractor = self._make_time_extractor(self.era5.coord_dim)
        else:
            time_extractor = self._make_time_extractor_explicit(self.era5.coord_dim, time)
        return self._with_obs_times(
            self.era5.get_data(time=time_extractor, lev_idx=lev_idx,
                               loc=loc, var_subset=var_subset),
            time is None)

    def get_grid_xarray(self) -> xr.Dataset:
        return self.sparse.get_grid_xarray()


    def to_xarray(self,
                  loc : Optional[Union[int, List[int]]] = None
                  ) -> xr.Dataset:
        """Convert to xarray Dataset
        """

        if loc is not None and isinstance(loc, int):
            loc = [loc]

        lead_dim = self.obs_index_dim

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



    def compute_p_obs(self):
        """Calculate the pressure at observation altitude via ERA5 vertical interpolation.

        Altitude at each ERA5 pressure level is derived from the hypsometric
        equation (cumulative sum from the surface).  ERA5 plev is stored in
        ascending pressure order (1 → 1000 hPa), i.e. descending altitude, so
        z_lev is descending along plev.  The arrays are flipped before being
        passed to regrid_pressure, which requires ascending p_src.
        """
        from src.physics.constants import CONST_G, CONST_Rd
        from src.physics.optics import regrid_pressure

        # get_era5() co-locates to each observation (coord + time) → (plev, time)
        era5_obs = self.get_era5(lev_idx=None, var_subset=["t", "sp", "z_sfc"])
        # Drop the obs-time coordinate: it must not propagate through the
        # hypsometric computation and conflict with model-time coords downstream.
        era5_obs = era5_obs.drop_vars("time", errors="ignore")
        for var in ["t", "sp", "z_sfc"]:
            assert var in era5_obs, f"ERA5 data must contain '{var}' for compute_p_obs"

        vert_dim = self.era5.vert_dim
        assert vert_dim is not None
        assert vert_dim in era5_obs.dims, f"ERA5 data must contain vertical dimension '{vert_dim}' for compute_p_obs"

        sp    = era5_obs["sp"]
        z_sfc = era5_obs["z_sfc"] / CONST_G           # model surface altitude (m)
        t     = era5_obs["t"]                         # temperature at each level
        obs_dim = self.obs_index_dim
        alt_obs_da = xr.DataArray(
            self.sparse.getv("alt"), dims=[obs_dim],
            coords={obs_dim: era5_obs[obs_dim]}
        )

        # When the model orography is above the observation, extrapolate sp downward
        # to the observation altitude.  Leave sp unchanged otherwise.
        t_sfc = t.isel({vert_dim: -1})  # near-surface temperature (1000 hPa level)
        sp_corrected = xr.where(
            z_sfc > alt_obs_da,
            sp * np.exp(CONST_G * (z_sfc - alt_obs_da) / (CONST_Rd * t_sfc)),
            sp,
        )

        # Pressure at each level clipped to the corrected surface pressure
        # (sub-surface levels collapse to sp_corrected, giving dz = 0 there)
        p_lev = era5_obs[vert_dim].clip(max=sp_corrected)

        vax = p_lev.dims.index(vert_dim)  # axis of vert_dim after broadcasting

        # Estimate pressure at half levels
        p_half = np.concatenate(
            [
                p_lev.isel({vert_dim: [0]}).values / 2,
                0.5 * (p_lev.isel({vert_dim: slice(None, -1)}).values +
                       p_lev.isel({vert_dim: slice(1, None)}).values),
                p_lev.isel({vert_dim: [-1]}).values,
            ],
            axis=vax,
        )
        dlogp = xr.DataArray(
            data=np.diff(np.log(p_half), axis=vax),
            coords=p_lev.coords,
            dims=p_lev.dims,
        )
        # Positive layer altitude thickness: (Rd/g) * T * d(ln p) > 0
        pos_dz = (CONST_Rd / CONST_G) * t * dlogp

        # Altitude at level i = alt_obs + sum of layer thicknesses from i to surface.
        # Anchoring to alt_obs (not z_sfc) ensures the column is always aligned
        # with the actual observation, regardless of model terrain offset.
        # Reverse cumsum: flip → cumsum → flip back.
        flip = {vert_dim: slice(None, None, -1)}
        z_lev = alt_obs_da + pos_dz.isel(flip).cumsum(dim=vert_dim).isel(flip)
        # z_lev is descending along plev (high altitude at index 0 = 1 hPa).
        z_lev = z_lev.isel(flip)
        p_lev = p_lev.isel(flip)

        # Each observation its altitude
        obs_alt = alt_obs_da

        # Interpolate pressure as a function of altitude → pressure at obs altitude
        return regrid_pressure(
            field_src=p_lev,
            p_src=z_lev,
            p_tgt=obs_alt.expand_dims("_z_obs"),
            src_vdim=vert_dim,
            tgt_vdim="_z_obs",
        ).squeeze("_z_obs")


class INPCollection(ObsCollection):
    """Subclass for INP data"""

    _own_dense_values: List[str] = ["INP_conc"]
    _own_dense_dtypes: Dict = {}
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

        super().initialise(non_index_attrs=self.dense_values,
                           extra_sparse_values=self.sparse_values)

        if inpdb_path is not None:
            print(f"Loading INP collection from {inpdb_path}")
            self.load_inp_collection(inpdb_path, n_lines,
                                     lonlat_approx=lonlat_approx,
                                     lon_to_degeast=lon_to_degeast,
                                     t_approx_h=t_approx_h, sortbytime=sortbytime,
                                     timerange=timerange)
        else:
            return

        if cams_clim_path is not None and cams_clim_path != "":
            print(f"Loading CAMS climatology from {cams_clim_path}")
            self.load_cams_clim(cams_clim_path)

        if cams_free_path is not None and cams_free_path != "":
            print(f"Loading CAMS free output from {cams_free_path}")
            self.load_cams_free(cams_free_path)

        self.era5_points_file = get_era5_inpdb_path()
        if self.era5_points_file != "" and self.era5_points_file is not None:
            print(f"Loading ERA5 output from {self.era5_points_file}")
            self.load_era5(self.era5_points_file)

            self.era5_points_geopot_file = get_era5_inpdb_path(geopot=True)
            print(f"Loading ERA5 geopotential output from {self.era5_points_geopot_file}")
            self.load_era5_geopot(self.era5_points_geopot_file)


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

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        #self.instruments.extend(df['Instrument'].tolist())
        #self.instrument_types.extend(df['Instrument_Type'].tolist())

        # Build unique-value lists and per-row indexes via factorize
        sparse_col_map = {
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
        dense_col_map = {}
        inp_var_names = ["IN [L-1]", "INP_amb [L-1]"]
        for inp_var in inp_var_names:
            if inp_var in df.columns:
                dense_col_map = {
                    "INP_conc": df[inp_var].to_numpy(dtype=float)
                }
                break
        if "INP_conc" not in dense_col_map:
            raise ValueError(f"None of the expected INP concentration columns {inp_var_names} found in the CSV file.")

        if lonlat_approx is not None:
            for attr in ["lon", "lat"]:
                values = sparse_col_map[attr]
                sparse_col_map[attr] = np.round(values / lonlat_approx) * lonlat_approx

        if lon_to_degeast:
            sparse_col_map["lon"] = np.mod(sparse_col_map["lon"], 360)

        # Populate the sparse 1D index
        for attr, values in sparse_col_map.items():
            self.sparse.store(attr, values)

        # Populate coords
        self.sparse.set_coords()

        # The dense values (obs_id is handled separately below)
        for dense_attr, dense_vals in dense_col_map.items():
            getattr(self, dense_attr).values = dense_vals  # type: ignore[attr-defined]

        getattr(self, self.obs_index_dim).values = np.arange(len(self.sparse.getv("time")), dtype=np.int64)  # type: ignore[attr-defined]

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
        cams_data = self.get_cams_clim(loc=loc)
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
        cams_data = self.get_cams_free(loc=loc)
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

    def _make_time_extractor_explicit(self, coord_dim: str, time) -> xr.DataArray:
        """Expands to ``(obs_id, time)``: each observation's fixed location is paired
        with all requested times, giving ``(obs_id, time[, lev])`` output.
        """
        if isinstance(time, xr.DataArray):
            return time
        obs_dim = self.obs_index_dim
        obs_ids = getattr(self, obs_dim).values  # type: ignore[attr-defined]
        time_arr = np.atleast_1d(np.asarray(time))
        return xr.DataArray(
            data=np.broadcast_to(time_arr[np.newaxis, :], (len(obs_ids), len(time_arr))),
            dims=[obs_dim, "time"],
            coords={
                obs_dim: obs_ids,
                "time": time_arr,
                coord_dim: (obs_dim, self.sparse.coord.index)
            }
        )


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
        self.handlers  : List[str] = ["cams_clim", "cams_free", "era5",
                                       "macv2sp", "macv2nat", "merra2"]
        self.cams_clim = CamsClimHandler(vert_dim="lev")
        self.cams_free = CamsOutputHandler(vert_dim="plev")
        self.era5 = ERA5DataHandler(vert_dim="plev")
        self.macv2sp = MacV2SPHandler(vert_dim=None)
        self.macv2nat  : MacV2NatHandler = MacV2NatHandler(vert_dim=None)
        self.merra2 = MERRA2Handler(vert_dim=None)

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
        for handler_name in self.handlers:
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                if isinstance(handler, ModelHandler):
                    subset_handler = cast(ModelHandler, handler.get_subset(**get_subset_kwargs))
                    setattr(subset, handler_name, subset_handler)

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
        lon_west = lon_west % 360.0
        lon_east = lon_east % 360.0
        def _lon_in_region(lon: float) -> bool:
            if lon_west <= lon_east:
                return lon_west <= lon <= lon_east
            else:
                return lon >= lon_west or lon <= lon_east
        stations_in_region = [name for name, station in self.stations.items()
                              if lat_south <= station.lat <= lat_north and _lon_in_region(station.lon % 360.0)]
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

        if self.era5.data is not None and era5_geopot.data is not None:
            self.era5.data["z_sfc"] = era5_geopot.data["z"].squeeze(drop=True)

    def add_cams_clim(self, clim_path: str,
                      cams_clim_id : str, lev_dim : str = "lev") -> None:
        """Attach a CamsClimHandler under a custom attribute name and register it in self.handlers."""
        if hasattr(self, cams_clim_id):
            raise ValueError(f"Attribute '{cams_clim_id}' already exists in StationsCollection.")
        cams_clim_handler = CamsClimHandler(vert_dim=lev_dim)
        cams_clim_handler.from_ncfile(clim_path)
        cams_clim_handler = cast(CamsClimHandler, cams_clim_handler.get_subset(
            coord_values=np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol))
        self._swap_model_dim_to_station_name(cams_clim_handler)
        setattr(self, cams_clim_id, cams_clim_handler)
        self.handlers.append(cams_clim_id)

    def load_macv2nat(self, macv2nat_path: str, ref_year : int = 2005) -> None:
        """Load MACv2 natural aerosol climatology and align to station coordinates.
        MACv2-SP must be loaded first (wavelength axis is taken from it).
        """
        assert self.macv2sp.data is not None, "Load MACv2-SP data before MACv2-NAT."

        open_kwargs : dict = {}
        if macv2nat_path.endswith(".nc"):
            open_kwargs["engine"] = "netcdf4"
        else:
            open_kwargs["engine"] = "zarr"
        macv2nat_ds = xr.open_dataset(macv2nat_path, **open_kwargs)  # type: ignore

        required_wls = np.sort(self.macv2sp.data["wavelength"].values)
        available_wls = np.sort(macv2nat_ds["wavelength"].values)
        missing_wls = [wl for wl in required_wls if wl not in available_wls]
        if missing_wls:
            macv2nat_ds = xr.concat([
                macv2nat_ds,
                xr.full_like(
                    macv2nat_ds.isel(wavelength=[0] * len(missing_wls)),
                    fill_value=np.nan
                ).assign_coords({"wavelength": missing_wls})],
                dim="wavelength")
            aod_da = macv2nat_ds["aod"]
            ssa_da = macv2nat_ds["ssa"]
            asy_da = macv2nat_ds["asy"]
            from ..physics.optics import angstrom_fill_nans
            for wl_tgt in required_wls:
                angstrom_fill_nans(aod_da, wl_tgt, wl_dim="wavelength")
                angstrom_fill_nans(ssa_da, wl_tgt, wl_dim="wavelength")
                angstrom_fill_nans(asy_da, wl_tgt, wl_dim="wavelength")
        else:
            aod_da = macv2nat_ds["aod"]
            ssa_da = macv2nat_ds["ssa"]
            asy_da = macv2nat_ds["asy"]

        self.macv2nat.data = xr.Dataset({
            "aod": aod_da,
            "ssa": ssa_da,
            "asy": asy_da,
        }).sel(wavelength=required_wls)
        self.macv2nat = cast(MacV2NatHandler, self.macv2nat.get_subset(
            coord_values=np.asarray(self.stations_coords),
            coords_to_assign={"station_name": self.station_names},
            atol=self.coord_atol))
        self._swap_model_dim_to_station_name(self.macv2nat)

    def get_macv2nat(self,
                     time : Union[np.datetime64, List, np.ndarray, xr.DataArray],
                     loc : Union[None, int, List[int], xr.DataArray] = None,
                     var_subset : Union[None, str, List[str]] = None) -> xr.Dataset:
        """Get MACv2 natural aerosol climatology at requested times."""
        if isinstance(time, np.datetime64):
            time = [time]
        if not isinstance(time, xr.DataArray):
            time_extractor = xr.DataArray(data=time, dims=["time"], coords={"time": time})
        else:
            time_extractor = time
        return self.macv2nat.get_data(time=time_extractor, loc=loc, var_subset=var_subset)

    def get_merra2(self,
                   time : Union[np.datetime64,
                                List[np.datetime64],
                                np.ndarray,
                                xr.DataArray],
                   loc : Union[None, int, List[int], xr.DataArray] = None,
                   var_subset : Union[None, str, List[str]] = None
                   ) -> xr.Dataset:
        """Get MERRA2 data at requested times."""

        assert self.merra2.data is not None, "MERRA2 data handler is not loaded yet!"

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

        assert self.macv2sp.data is not None, "MACv2-SP data handler is not loaded yet!"

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
        """Get CAMS climatology interpolated to times.
        """

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

    cams_clim_points_file : str = get_cams_clim_aeronet_path(data_product="AOD")
    cams_free_points_file : str = get_cams_free_aeronet_path(data_product="AOD")
    #macv2sp_points_file : str = get_macv2sp_aeronet_path()

    def __init__(self, aeronet_dir_path : Optional[str] = None,
                 cams_clim_path : Optional[str] = None,
                 cams_free_path : Union[None, str, List[str]] = None,
                 macv2sp_path : Optional[str] = None,
                 macv2nat_path : Optional[str] = None,
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

        if aeronet_dir_path is None:
            return


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

        if macv2nat_path != "" and macv2nat_path is not None:
            print(f"Loading MACv2 natural climatology from {macv2nat_path}")
            self.load_macv2nat(macv2nat_path)

        if merra2_path != "" and merra2_path is not None:
            print(f"Loading MERRA2 from {merra2_path}")
            # try:
            self.load_merra2(merra2_path)
            # except Exception as exc:
            #     print(f"Warning: Failed to load MERRA2 from {merra2_path}: {exc}")

        if self.era5_points_file != "" and self.era5_points_file is not None:
            print(f"Loading ERA5 output from {self.era5_points_file}")
            # try:
            self.load_era5(self.era5_points_file)
            # except Exception as exc:
            #     print(f"Warning: Failed to load ERA5 output from {self.era5_points_file}: {exc}")

            self.era5_points_geopot_file = get_era5_aeronet_path(aeronet_freq=aeronet_freq,
                                                                 aeronet_product="AOD",
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
            print(f"Looking under {aeronet_dir_path}")
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
            df_merged.to_parquet(synth_file)
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
                    angstrom_fill : bool = True,
                    aod_wl: Optional[List[int]] = None,
                    ae_wl: Optional[List[Tuple[int, int]]] = None,
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
        aewl_tags = {f"{wl[0]}-{wl[1]}_Angstrom_Exponent" for wl in (ae_wl or self.ae_wl)}
        ae_cols = [c for c in df.columns if c in aewl_tags]

        extra_cols: List[str] = []
        if include_precipitable_water and "Precipitable_Water(cm)" in df.columns:
            extra_cols.append("Precipitable_Water(cm)")

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
        target_wl  = np.asarray(aod_wl or self.aod_wl, dtype=int)
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

        if ae_cols != []:
            ds["AE"] = xr.concat(
                [ds[ae_var].expand_dims(
                    wavelength_ae_nm=["-".join(re.search(r"(\d+)-(\d+)_Angstrom_Exponent", ae_var).groups())]  # type: ignore
                ) for ae_var in ae_cols],
                dim="wavelength_ae_nm",
            )
            ds = ds.drop_vars(aod_cols + ae_cols)  # drop original columns

        return ds

    _AODSources = Literal["cams_clim", "cams_clim_natural", "cams_free",
                          "macv2sp+cams", "macv2sp", "merra2", "macv2nat",
                          "volc_glossac"]
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
                      skip_ae: bool = True,
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
            assert self.aod_wl is not None, "AOD wavelengths not set."
            wavelengths_nm = tuple(sorted(set(self.aod_wl)))

        # Ensure ae wl in wavelengths_nm when AE is requested
        if not skip_ae and self.ae_wl is not None:
            all_ae_wls: List[int] = []
            for wl_couple in self.ae_wl:
                all_ae_wls.extend(wl_couple)
            wavelengths_nm_tmp = tuple(sorted(set(list(wavelengths_nm) + all_ae_wls)))
        else:
            wavelengths_nm_tmp = wavelengths_nm

        if times is None:
            time_key = None
            assert self.era5.data is not None
            times = self.era5.data["time"]
        else:
            tvals = np.asarray(getattr(times, "values", times), dtype="datetime64[ns]")
            time_key = (str(tvals.min()), str(tvals.max()), int(tvals.size))
        cache_key = (source, chunk,
                     tuple(wavelengths_nm), bool(per_species),
                     bool(skip_ae),
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
            handler_ds = self.get_cams_free(time=times, lev_idx=None)  # all levels
            # And AE would be missing?
            var_list = []
            missing_wls = []
            for wl in wavelengths_nm_tmp:
                var_name = f"aod{wl:d}"
                if var_name in handler_ds.data_vars:
                    var_list.append(handler_ds[var_name].expand_dims(wavelength=[wl]))
                else:
                    missing_wls.append(wl)
            if missing_wls:
                if not var_list:
                    raise ValueError(f"No CAMS free AOD data available for wavelengths {missing_wls}")
                var_list.extend([xr.full_like(var_list[0], None).assign_coords(wavelength=[wl]) for wl in missing_wls])
            aod_da = xr.concat(var_list, dim="wavelength").sortby("wavelength")

            aod_ds = xr.Dataset(data_vars={"AOD": aod_da})

            # Angstrom fill missing values
            if missing_wls:
                print(f"Angstrom fill of missing wls: {missing_wls}")
                print("not implemented yet")
                pass
            if not skip_ae:
                print(f"No AE calculation yet implemented")
                pass

            # TODO: if AERONET data are monthly
            # do monthly means of all CAMS available timesteps
            print("Warning: CAMS free AOD not yet averaged to monthly means, returning instantaneous values")
            return aod_ds.sel(wavelength=list(wavelengths_nm))


        elif source == "macv2sp+cams":
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
            result["AOD_tropo"] = result["AOD_tropo"] + result["AOD_anthropogenic"]
            result["AOD"] = result["AOD_natural"] + result["AOD_anthropogenic"]

            self._aod_cache[cache_key] = result

            return self._aod_cache[cache_key]
        elif source == "macv2sp":
            from src.physics.volcaero import get_glossac_aod
            result = self.calculate_aod(
                times=times, source="macv2nat", chunk=chunk,
                wavelengths_nm=wavelengths_nm, per_species=False,
                aero_opt_ver=aero_opt_ver, optics_lut_path=optics_lut_path,
                clip_z_to_station=clip_z_to_station, recompute=recompute,
                n_workers=n_workers, show_progress=show_progress,
            )
            result["AOD_anthropogenic"] = (self.get_macv2sp(time=times)["aod_2D"]
                                           .sortby("wavelength")
                                           .sel(wavelength=list(wavelengths_nm), method="nearest"))
            result["AOD_tropo"] = result["AOD_tropo"] + result["AOD_anthropogenic"]
            result["AOD"] = result["AOD_natural"] + result["AOD_anthropogenic"]
            self._aod_cache[cache_key] = result
            return result

        elif source == "macv2nat":
            from src.physics.volcaero import get_glossac_aod
            aod_natural = (self.get_macv2nat(time=times)["aod"]
                         .rename("AOD_natural")
                         .sortby("wavelength")
                         .sel(wavelength=list(wavelengths_nm), method="nearest")
                         .sortby("wavelength")
                         )
            result = xr.Dataset(data_vars={"AOD_tropo": aod_natural})
            result["AOD_volc_glossac"] = (get_glossac_aod()
                .interp(lat=result.lat, method="linear")
                .rename(wavelength_nm="wavelength")
                .sel(wavelength=list(wavelengths_nm), method="nearest")
                .sel(time=result.time, method="nearest")
                ).sortby("wavelength")
            result["AOD_natural"] = result["AOD_tropo"] + result["AOD_volc_glossac"]
            self._aod_cache[cache_key] = result
            return result

        elif source == "volc_glossac":
            from src.physics.volcaero import get_glossac_aod
            assert self.era5.data is not None
            glossac_aod = (get_glossac_aod()
                .interp(lat=self.era5.data.lat, method="linear")
                .sel(wavelength_nm=list(wavelengths_nm), method="nearest"))
            result = xr.Dataset(data_vars={"AOD_volc_glossac": glossac_aod})
            self._aod_cache[cache_key] = result
            return result

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
                                  show_progress=show_progress,
                                  wavelengths_nm=list(wavelengths_nm_tmp))
        # Add accessory coordinates from era5
        era5_coords = self.era5.data.coords
        for coord in era5_coords:
            coord_dims = era5_coords[coord].dims
            if len(coord_dims) == 1 and coord_dims[0] in result.dims and coord not in result.coords:
                result = result.assign_coords({coord: era5_coords[coord]})

        from src.physics.volcaero import get_glossac_aod
        try:
            print("Adding volcanic aerosol AOD from GloSSAC dataset")
            glossac_aod = get_glossac_aod().interp(lat=result.lat, method="linear").rename(wavelength_nm="wavelength").sel(wavelength=result.wavelength, method="nearest")
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

        if not skip_ae:
            result["AE"] = xr.concat(
                [(np.log(result.sel(wavelength=w[0])["AOD"] /
                        result.sel(wavelength=w[1])["AOD"]) /
                np.log(w[1] / w[0])).expand_dims(wavelength_ae=[f"{w[0]}-{w[1]}"])
                for w in self.ae_wl],
                dim="wavelength_ae",
            )
            result = result.sel(wavelength=wavelengths_nm).sortby("wavelength").sortby("wavelength_ae")

        self._aod_cache[cache_key] = result

        return self._aod_cache[cache_key]
