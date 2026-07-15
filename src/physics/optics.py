"""Offline aerosol optics / AOD computation.


Notes
-----
* ERA5 relative humidity is stored in percent (0-100) while the optics LUT RH
  axis is fraction (0-1);
* Layer mass is approximated as ``|dp| / g`` with ``dp`, top/bottom half-layers are slightly
  misrepresented not a big issue.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import xarray as xr

from . import vinterp
from .aerosol import OPTICS_AERO_MAP

DEFAULT_OPTICS_LUT = "/home/papa/data/aerosol_ifs_49R1_20230725.nc"
DEFAULT_AERO_OPT_VER = "48r1_4dclim"

G = 9.80665  # m s-2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _with_dim_coords(da: xr.DataArray) -> xr.DataArray:
    """Assign default integer coordinates to any dimension lacking one.

    ``tools_to_stack_xarrays`` reads ``arr.coords[dim]`` for every non-intp
    dimension, so dimensions without a coordinate (e.g. ``col``) would raise a
    ``KeyError``.  The added coordinates are harmless and consistent across
    arrays sharing a dimension (same size -> same ``arange``).
    """
    missing = {d: np.arange(da.sizes[d]) for d in da.dims if d not in da.coords}
    return da.assign_coords(missing) if missing else da


def regrid_pressure(
    field_src: xr.DataArray,
    p_src: xr.DataArray,
    p_tgt: xr.DataArray,
    *,
    src_vdim: str,
    tgt_vdim: str,
    intp_name: str = "_p_intp",
) -> xr.DataArray:
    """Interpolate ``field_src`` from its ``p_src`` levels onto ``p_tgt`` levels.

    The interpolation is performed in pressure using the Fortran routine.
    ``p_src`` must be ascending along ``src_vdim``.

    Parameters
    ----------
    field_src
        Field defined on the source vertical grid (dim ``src_vdim``).
    p_src
        Source pressure along ``src_vdim`` (ascending).  May carry the same
        broadcast dimensions as ``field_src`` or just ``src_vdim``.
    p_tgt
        Target pressure along ``tgt_vdim``.
    src_vdim, tgt_vdim
        Names of the source / target vertical dimensions.
    """
    ps = _with_dim_coords(p_src.rename({src_vdim: intp_name}))
    pt = _with_dim_coords(p_tgt.rename({tgt_vdim: intp_name}))

    st = vinterp.tools_to_stack_xarrays(src_arr=ps, dst_arr=pt, intp_dim_name=intp_name)
    tgtlevs, weights = vinterp.interp(
        psrc=ps.transpose(*st.src_dim_order).values.reshape(st.src_stackshape),
        ptgt=pt.transpose(*st.dst_dim_order).values.reshape(st.dst_stackshape),
    )
    tgtlevs = xr.DataArray(tgtlevs.reshape(st.out_shape), dims=st.out_dim_order, coords=st.out_coords)
    weights = xr.DataArray(weights.reshape(st.out_shape), dims=st.out_dim_order, coords=st.out_coords)

    fs = _with_dim_coords(field_src.rename({src_vdim: intp_name}))
    st2 = vinterp.tools_to_stack_xarrays(src_arr=fs, dst_arr=tgtlevs, intp_dim_name=intp_name)
    out = vinterp.interp_fld(
        fsrc=fs.transpose(*st2.src_dim_order).values.reshape(st2.src_stackshape),
        tgtlevs=tgtlevs.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
        weights=weights.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
    )
    out = xr.DataArray(out.reshape(st2.out_shape), dims=st2.out_dim_order, coords=st2.out_coords)
    return out.rename({intp_name: tgt_vdim})


def interp_optics_to_rh(
    optics_lut: xr.Dataset,
    rh_field: xr.DataArray,
    *,
    intp_name: str = "_rh_intp",
) -> xr.Dataset:
    """Interpolate an optics LUT onto a target relative-humidity field.

    ``optics_lut`` must expose a ``relative_humidity`` variable (ascending,
    fraction 0-1) along a ``relative_humidity`` dimension.  ``rh_field`` carries
    the *target* RH (fraction) and must **not** have a ``relative_humidity``
    dimension.  Variables without a ``relative_humidity`` dependence are passed
    through unchanged.
    """
    src = _with_dim_coords(optics_lut["relative_humidity"].rename({"relative_humidity": intp_name}))
    dst = _with_dim_coords(rh_field).expand_dims({intp_name: [0.0]}, axis=-1)

    st = vinterp.tools_to_stack_xarrays(src_arr=src, dst_arr=dst, intp_dim_name=intp_name)
    tgtlevs, weights = vinterp.interp(
        psrc=src.transpose(*st.src_dim_order).values.reshape(st.src_stackshape),
        ptgt=dst.transpose(*st.dst_dim_order).values.reshape(st.dst_stackshape),
    )
    tgtlevs = xr.DataArray(tgtlevs.reshape(st.out_shape), dims=st.out_dim_order, coords=st.out_coords)
    weights = xr.DataArray(weights.reshape(st.out_shape), dims=st.out_dim_order, coords=st.out_coords)

    out_vars = []
    for name in optics_lut.data_vars:
        if name == "relative_humidity":
            continue
        da = optics_lut[name]
        if "relative_humidity" not in da.dims:
            out_vars.append(da)
            continue
        fs = da.drop_vars([c for c in da.coords if c not in da.dims], errors="ignore")
        fs = _with_dim_coords(fs.rename({"relative_humidity": intp_name}))
        st2 = vinterp.tools_to_stack_xarrays(src_arr=fs, dst_arr=tgtlevs, intp_dim_name=intp_name)
        res = vinterp.interp_fld(
            fsrc=fs.transpose(*st2.src_dim_order).values.reshape(st2.src_stackshape),
            tgtlevs=tgtlevs.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
            weights=weights.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
        )
        res = xr.DataArray(res.reshape(st2.out_shape), dims=st2.out_dim_order, coords=st2.out_coords)
        out_vars.append(res.squeeze(intp_name, drop=True).rename(name))

    return xr.merge(out_vars)


# -----------------------------------------------------------------------------
# AOD calculator
# -----------------------------------------------------------------------------
class AODCalculator:
    """Compute offline aerosol optical depth from mmr fields + relative humidity.

    The optics LUT is interpolated to the wavelengths requested at construction
    time and (for hydrophilic species) to the relative humidity carried by the
    input fields.  Wavelength interpolation of the LUT is cheap (scalar targets)
    and cached per species; RH and vertical interpolation go through the Fortran
    interface.

    Parameters
    ----------
    optics_lut_path
        Path to the IFS aerosol optics NetCDF LUT.
    aero_opt_ver
        Key into :data:`src.physics.aerosol.OPTICS_AERO_MAP` mapping species
        name -> ``(optics_index, is_hydrophilic)``.
    wavelengths_nm
        Wavelengths (nm) at which AOD is evaluated.
    """

    def __init__(
        self,
        optics_lut_path: Optional[str] = None,
        aero_opt_ver: str = DEFAULT_AERO_OPT_VER,
        wavelengths_nm: Sequence[float] = (550.0,),
    ) -> None:
        self.optics_lut_path = optics_lut_path or DEFAULT_OPTICS_LUT
        self.aero_opt_ver = aero_opt_ver
        self.wavelengths_nm = tuple(float(w) for w in wavelengths_nm)
        self.g = G

        self._optics_lut: Optional[xr.Dataset] = None
        self._optics_cache: dict = {}

    # -- lazily loaded data ---------------------------------------------------
    @property
    def optics_lut(self) -> xr.Dataset:
        if self._optics_lut is None:
            lut = xr.load_dataset(self.optics_lut_path)
            lut["relative_humidity"] = (lut["relative_humidity1"] + lut["relative_humidity2"]) / 2
            self._optics_lut = lut
        return self._optics_lut

    @property
    def aeromap(self) -> dict:
        return OPTICS_AERO_MAP[self.aero_opt_ver]

    # -- optics interpolation -------------------------------------------------
    def _species_optics(
        self,
        spec: str,
        opt_idx: int,
        is_philic: bool,
        wns: np.ndarray,
        wls: Sequence[float],
    ) -> xr.Dataset:
        """Wavelength-interpolated optics slice for one species (cached)."""
        key = (spec, bool(is_philic), tuple(wls))
        if key in self._optics_cache:
            return self._optics_cache[key]

        lut = self.optics_lut
        if is_philic:
            sub = lut[["mass_ext_hydrophilic", "ssa_hydrophilic", "relative_humidity"]].isel(
                hydrophilic=opt_idx - 1
            )
        else:
            sub = lut[["mass_ext_hydrophobic", "ssa_hydrophobic"]].isel(hydrophobic=opt_idx - 1)

        sub = sub.interp(wavenumber=list(np.asarray(wns)), method="linear")
        sub = sub.assign_coords(
            wavelength=xr.DataArray(data=np.asarray(list(wls), dtype=int), dims=["wavenumber"])
        ).swap_dims(wavenumber="wavelength")
        sub = sub.drop_vars("wavenumber", errors="ignore")

        self._optics_cache[key] = sub
        return sub

    # -- vertical layer mass --------------------------------------------------
    def _layer_mass(self, pressure: xr.DataArray, lev_dim: str = "lev") -> xr.DataArray:
        """Approximate per-level air mass ``|dp| / g`` (kg m-2)."""
        ax = pressure.get_axis_num(lev_dim)
        dp = np.abs(np.gradient(pressure.values, axis=ax))
        return xr.DataArray(dp / self.g, dims=pressure.dims, coords=pressure.coords)

    # -- time grouping --------------------------------------------------------
    @staticmethod
    def _time_groups(times: xr.DataArray, chunk: str) -> List[xr.DataArray]:
        vals = times.values
        if chunk == "year":
            keys = times.dt.year.values
        elif chunk == "month":
            keys = times.dt.year.values * 100 + times.dt.month.values
        else:
            raise ValueError("chunk must be 'year' or 'month'")

        groups = []
        for k in np.unique(keys):
            sel = vals[keys == k]
            groups.append(xr.DataArray(sel, dims="time", coords={"time": sel}))
        return groups

    # -- aerosol fields at target times --------------------------------------
    def _aero_at(self, aero: xr.Dataset, tt: xr.DataArray) -> Union[xr.Dataset, xr.DataArray]:
        """Return aerosol fields at the target times ``tt``.

        If ``aero`` is a monthly climatology (``month`` dim), it is
        time-interpolated; if it already has a ``time`` dim it is sliced.
        """
        if "time" in aero.dims:
            return aero.sel(time=tt.values).load()
        if "month" in aero.dims:
            from ..utils.cams import interpolate_monthly_clim  # noqa: PLC0415

            return interpolate_monthly_clim(aero, tt).load()
        raise ValueError("`aero` must have a 'time' or 'month' dimension")

    # -- main entry point -----------------------------------------------------
    def compute_aod(
        self,
        aero: xr.Dataset,
        rh: Union[xr.Dataset, xr.DataArray],
        *,
        target_times: Optional[Union[np.ndarray, xr.DataArray]] = None,
        chunk: str = "year",
        wavelengths_nm: Optional[Sequence[float]] = None,
        sum_levels: bool = True,
        per_species: bool = False,
        lev_dim: str = "lev",
        rh_vert_dim: str = "plev",
        rh_var: str = "r",
        rh_pressure_var: str = "p",
        rh_pressure: Optional[xr.DataArray] = None,
        surface_pressure: Optional[xr.DataArray] = None,
        rh_in_percent: bool = True,
        n_workers: int = 1,
        show_progress: bool = False,
    ) -> xr.Dataset:
        """Compute AOD (and absorption) from aerosol mmr fields + relative humidity.

        For every target timestep the nearest-in-time ``rh`` profile is selected
        on its native (finer) ERA5 pressure levels, the aerosol mmr fields are
        interpolated vertically up onto those levels, the RH is used to look up
        the optics LUT, and the layer air mass on the ERA5 levels gives the AOD.

        Parameters
        ----------
        aero
            Aerosol mass mixing ratios (kg/kg) plus a ``pressure`` variable.
            Either already on a ``time`` grid, or a monthly climatology with a
            ``month`` (and optionally ``epoch``) dimension, in which case it is
            time-interpolated per chunk.
        rh
            Relative humidity field, or a dataset containing it (``rh_var``)
            and -- when ERA5 is on model levels -- a pressure variable
            (``rh_pressure_var``).
        rh_pressure
            Explicit target pressure on ``rh_vert_dim`` (overrides any pressure
            found in ``rh``).  When neither is available the ``rh_vert_dim``
            coordinate is used as the pressure, which is correct only when ERA5
            is on pressure levels.
        surface_pressure
            Station surface pressure (Pa).  When given, the target pressure is
            clipped to it so that only the layer thickness *above* the station
            contributes to the column integral (levels below ground get zero
            layer mass).
        target_times
            Times to evaluate.  Defaults to ``aero['time']`` when available.
        chunk
            ``"year"`` (default) groups the computation per calendar year;
            ``"month"`` splits it into per-month computations (lower memory).
        wavelengths_nm
            Override the construction-time wavelengths.
        sum_levels
            Sum the column (return AOD) instead of per-level extinction.
        per_species
            Also return per-species ``AOD_<spec>`` / ``Abs_<spec>``.
        n_workers
            Number of parallel workers for chunk processing.  ``1`` (default)
            runs sequentially.  ``>1`` uses a ``ThreadPoolExecutor``; zarr I/O,
            Fortran ctypes calls, and numpy arithmetic all release the GIL so
            threads give real parallelism for this workload without the
            fork-safety issues of ``ProcessPoolExecutor`` in notebooks.
        show_progress
            Show a ``tqdm`` progress bar over chunks.  Defaults to ``False``.
        """
        wls = list(self.wavelengths_nm if wavelengths_nm is None else [float(w) for w in wavelengths_nm])
        wns = 1.0e7 / np.asarray(wls, dtype=float)
        aeromap = self.aeromap

        if isinstance(rh, xr.Dataset):
            if rh_pressure is None and rh_pressure_var in rh.variables:
                rh_pressure = rh[rh_pressure_var]
            rh = rh[rh_var]

        if target_times is None:
            if "time" in aero.dims:
                target_times = aero["time"]
            else:
                raise ValueError("`target_times` must be provided when `aero` has no time dim")
        tvals = np.asarray(getattr(target_times, "values", target_times), dtype="datetime64[ns]")
        target_times = xr.DataArray(tvals, dims="time", coords={"time": tvals})

        groups = list(self._time_groups(target_times, chunk))

        # Pre-slice aero and rh per chunk in the main thread.
        # pandas.Index._cache is written lazily on first _index_as_unique access
        # and is not thread-safe; pre-selecting here gives each thread its own
        # independent Index object so no shared mutable state is touched.
        _aero_slices = (
            [aero.sel(time=tt.values) for tt in groups]
            if "time" in aero.dims
            else [aero] * len(groups)
        )
        _rh_slices = [
            rh.sel(time=tt.values, method="nearest").assign_coords(time=tt.values)
            for tt in groups
        ]
        chunk_args = [
            (a_slice, r_slice, tt, wls, wns, aeromap, sum_levels, per_species,
             lev_dim, rh_vert_dim, rh_in_percent, rh_pressure, surface_pressure)
            for a_slice, r_slice, tt in zip(_aero_slices, _rh_slices, groups)
        ]

        from tqdm.auto import tqdm
        n_chunks = len(chunk_args)

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            out_chunks: List[xr.Dataset] = [None] * n_chunks  # type: ignore[list-item]
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_idx = {
                    pool.submit(self._compute_chunk, *args): i
                    for i, args in enumerate(chunk_args)
                }
                with tqdm(total=n_chunks, desc="AOD chunks", disable=not show_progress) as pbar:
                    for future in as_completed(future_to_idx):
                        out_chunks[future_to_idx[future]] = future.result()
                        pbar.update()
        else:
            out_chunks = [
                self._compute_chunk(*args)
                for args in tqdm(chunk_args, desc="AOD chunks", disable=not show_progress)
            ]

        return xr.concat(out_chunks, dim="time")

    # -- per-chunk worker -----------------------------------------------------
    @staticmethod
    def _apply_rh_weights(
        optics_lut: xr.Dataset,
        tgtlevs: xr.DataArray,
        weights: xr.DataArray,
        intp_name: str = "_rh_intp",
    ) -> xr.Dataset:
        """Apply pre-computed RH interpolation weights to an optics LUT.

        Equivalent to ``interp_optics_to_rh`` but skips the ``vinterp.interp``
        call (weights already computed from the shared RH axis).
        """
        out_vars = []
        for name in optics_lut.data_vars:
            if name == "relative_humidity":
                continue
            da = optics_lut[name]
            if "relative_humidity" not in da.dims:
                out_vars.append(da)
                continue
            fs = da.drop_vars([c for c in da.coords if c not in da.dims], errors="ignore")
            fs = _with_dim_coords(fs.rename({"relative_humidity": intp_name}))
            st2 = vinterp.tools_to_stack_xarrays(src_arr=fs, dst_arr=tgtlevs, intp_dim_name=intp_name)
            res = vinterp.interp_fld(
                fsrc=fs.transpose(*st2.src_dim_order).values.reshape(st2.src_stackshape),
                tgtlevs=tgtlevs.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
                weights=weights.transpose(*st2.dst_dim_order).values.reshape(st2.dst_stackshape),
            )
            res = xr.DataArray(res.reshape(st2.out_shape), dims=st2.out_dim_order, coords=st2.out_coords)
            out_vars.append(res.squeeze(intp_name, drop=True).rename(name))
        return xr.merge(out_vars)

    def _compute_chunk(
        self,
        aero: xr.Dataset,
        rh: xr.DataArray,
        tt: xr.DataArray,
        wls: Sequence[float],
        wns: np.ndarray,
        aeromap: dict,
        sum_levels: bool,
        per_species: bool,
        lev_dim: str,
        rh_vert_dim: str,
        rh_in_percent: bool,
        rh_pressure: Optional[xr.DataArray] = None,
        surface_pressure: Optional[xr.DataArray] = None,
    ) -> xr.Dataset:
        a = self._aero_at(aero, tt)
        if "pressure" not in a:
            raise ValueError("`aero` must contain a 'pressure' variable for the AOD integration")

        # Nearest ERA5 relative humidity in time, aligned to the aerosol times.
        r = rh.sel(time=tt.values, method="nearest").assign_coords(time=tt.values)
        if rh_in_percent:
            r = r / 100.0
        r = r.clip(0.0, 1.0)

        # Integrate on the ERA5 pressure grid, which is finer than the
        # (climatology) aerosol grid.  RH stays on its native levels; the
        # aerosol mmr fields are interpolated up onto it and the layer mass is
        # taken from the ERA5 pressure levels.  The target pressure comes from an
        # explicit pressure field when ERA5 is on model levels (no pressure
        # coordinate), otherwise from the ``rh_vert_dim`` coordinate itself.
        if rh_pressure is not None:
            p_tgt = rh_pressure
            if "time" in p_tgt.dims:
                p_tgt = p_tgt.sel(time=tt.values, method="nearest").assign_coords(time=tt.values)
        else:
            p_tgt = r.coords[rh_vert_dim]

        # Clip the integration pressure to the station surface so that levels
        # below ground contribute no layer mass (only the thickness above the
        # station is non-zero).
        p_mass = p_tgt
        if surface_pressure is not None:
            sp_st = surface_pressure
            if "time" in sp_st.dims:
                sp_st = sp_st.sel(time=tt.values, method="nearest").assign_coords(time=tt.values)
            p_mass = xr.apply_ufunc(np.minimum, p_tgt, sp_st)
        layer_mass = self._layer_mass(p_mass, lev_dim=rh_vert_dim)

        species = [s for s in aeromap if s in a.data_vars and s != "pressure"]
        if not species:
            raise ValueError("No species in `aero` matched the optics map for this version")

        # --- Pressure interpolation weights (computed once, reused for every species) ---
        _p = "_p_intp"
        ps = _with_dim_coords(a["pressure"].rename({lev_dim: _p}))
        pt = _with_dim_coords(p_tgt.rename({rh_vert_dim: _p}))
        st_p = vinterp.tools_to_stack_xarrays(src_arr=ps, dst_arr=pt, intp_dim_name=_p)
        p_tgtlevs_raw, p_weights_raw = vinterp.interp(
            psrc=ps.transpose(*st_p.src_dim_order).values.reshape(st_p.src_stackshape),
            ptgt=pt.transpose(*st_p.dst_dim_order).values.reshape(st_p.dst_stackshape),
        )
        p_tgtlevs = xr.DataArray(
            p_tgtlevs_raw.reshape(st_p.out_shape), dims=st_p.out_dim_order, coords=st_p.out_coords
        )
        p_weights = xr.DataArray(
            p_weights_raw.reshape(st_p.out_shape), dims=st_p.out_dim_order, coords=st_p.out_coords
        )
        # Stacking info for field application — identical for all species (same field shape).
        fs0 = _with_dim_coords(a[species[0]].rename({lev_dim: _p}))
        st2_p = vinterp.tools_to_stack_xarrays(src_arr=fs0, dst_arr=p_tgtlevs, intp_dim_name=_p)
        p_tgtlevs_t = p_tgtlevs.transpose(*st2_p.dst_dim_order).values.reshape(st2_p.dst_stackshape)
        p_weights_t = p_weights.transpose(*st2_p.dst_dim_order).values.reshape(st2_p.dst_stackshape)

        # --- Batched RH interpolation for all hydrophilic species ---
        # All hydrophilic entries share the same LUT relative_humidity axis, so the
        # interpolation weights are identical.  Stack all species' LUT arrays along the
        # nsrc axis of interp_fld and do TWO Fortran calls total (mass_ext + ssa)
        # instead of 2 * n_philic calls.
        _rh = "_rh_intp"
        philic_specs = [s for s in species if aeromap[s][1]]
        philic_mass_ext: dict[str, np.ndarray] = {}
        philic_ssa: dict[str, np.ndarray] = {}

        if philic_specs:
            opt0 = self._species_optics(philic_specs[0], aeromap[philic_specs[0]][0], True, wns, wls)

            # Compute RH interp weights once from the shared LUT axis
            src_rh = _with_dim_coords(opt0["relative_humidity"].rename({"relative_humidity": _rh}))
            dst_rh = _with_dim_coords(r).expand_dims({_rh: [0.0]}, axis=-1)
            st_rh = vinterp.tools_to_stack_xarrays(src_arr=src_rh, dst_arr=dst_rh, intp_dim_name=_rh)
            rh_tgtlevs_raw, rh_weights_raw = vinterp.interp(
                psrc=src_rh.transpose(*st_rh.src_dim_order).values.reshape(st_rh.src_stackshape),
                ptgt=dst_rh.transpose(*st_rh.dst_dim_order).values.reshape(st_rh.dst_stackshape),
            )
            rh_tgtlevs = xr.DataArray(
                rh_tgtlevs_raw.reshape(st_rh.out_shape), dims=st_rh.out_dim_order, coords=st_rh.out_coords
            )
            rh_weights = xr.DataArray(
                rh_weights_raw.reshape(st_rh.out_shape), dims=st_rh.out_dim_order, coords=st_rh.out_coords
            )

            # Build stacking info once — all species share the same LUT shape (wavelength × rh)
            def _lut_src(da: xr.DataArray) -> np.ndarray:
                da = da.drop_vars([c for c in da.coords if c not in da.dims], errors="ignore")
                da = _with_dim_coords(da.rename({"relative_humidity": _rh}))
                return da.transpose(*st2_rh.src_dim_order).values.reshape(st2_rh.src_stackshape)

            da0_me = opt0["mass_ext_hydrophilic"]
            fs0_me = _with_dim_coords(
                da0_me.drop_vars([c for c in da0_me.coords if c not in da0_me.dims], errors="ignore")
                .rename({"relative_humidity": _rh})
            )
            st2_rh = vinterp.tools_to_stack_xarrays(src_arr=fs0_me, dst_arr=rh_tgtlevs, intp_dim_name=_rh)
            rh_tgt_t = rh_tgtlevs.transpose(*st2_rh.dst_dim_order).values.reshape(st2_rh.dst_stackshape)
            rh_wts_t = rh_weights.transpose(*st2_rh.dst_dim_order).values.reshape(st2_rh.dst_stackshape)
            rh_out_dims = [d for d in st2_rh.out_dim_order if d != _rh]
            rh_out_coords = {k: v for k, v in st2_rh.out_coords.items() if k != _rh}

            # Permutation to reorder philic arrays to canonical (wavelength, *st2_p spatial) order
            _canonical_philic = ("wavelength",) + tuple(
                d if d != _p else rh_vert_dim for d in st2_p.out_dim_order
            )
            _philic_perm = tuple(rh_out_dims.index(d) for d in _canonical_philic)

            # Stack all species along nsrc (wavelength) axis: (1, n_philic*n_wl, n_rh)
            me_src_list, ssa_src_list = [], []
            for s in philic_specs:
                opt_s = self._species_optics(s, aeromap[s][0], True, wns, wls)
                me_src_list.append(_lut_src(opt_s["mass_ext_hydrophilic"]))
                ssa_src_list.append(_lut_src(opt_s["ssa_hydrophilic"]))

            n_philic = len(philic_specs)
            me_stacked = np.concatenate(me_src_list, axis=1)    # (1, n_philic*n_wl, n_rh)
            ssa_stacked = np.concatenate(ssa_src_list, axis=1)  # (1, n_philic*n_wl, n_rh)

            # Two Fortran calls instead of 2*n_philic
            me_out = vinterp.interp_fld(fsrc=me_stacked, tgtlevs=rh_tgt_t, weights=rh_wts_t)
            ssa_out = vinterp.interp_fld(fsrc=ssa_stacked, tgtlevs=rh_tgt_t, weights=rh_wts_t)

            # Split result back into per-species numpy arrays: (n_philic, *st2_rh.out_shape)
            me_rs = me_out.squeeze(0).reshape(n_philic, *st2_rh.out_shape)
            ssa_rs = ssa_out.squeeze(0).reshape(n_philic, *st2_rh.out_shape)
            for i, s in enumerate(philic_specs):
                # squeeze(-1) drops the trailing _rh_intp size-1 dim; transpose to canonical order
                philic_mass_ext[s] = me_rs[i].squeeze(-1).transpose(_philic_perm)
                philic_ssa[s] = ssa_rs[i].squeeze(-1).transpose(_philic_perm)

        #
        # Numpy-only
        # Canonical spatial dim order: (time, station_name, plev) from st2_p
        spatial_dims = tuple(d if d != _p else rh_vert_dim for d in st2_p.out_dim_order)
        n_spatial = len(spatial_dims)
        lev_axis_in_output = spatial_dims.index(rh_vert_dim) + 1  # +1 for leading wavelength

        # Broadcast layer_mass to (1, n_time, n_station, n_plev) — pure numpy, no broadcast_like
        lm_perm = tuple(list(layer_mass.dims).index(d) for d in spatial_dims)
        lm_bc = layer_mass.values.transpose(lm_perm)[np.newaxis]  # (1, n_time, n_station, n_plev)

        # Precompute numpy axis-reorder for species fields (same shape for all species)
        _spec_dims = tuple(d if d != lev_dim else _p for d in a[species[0]].dims)
        _src_order = tuple(_spec_dims.index(d) for d in st2_p.src_dim_order)

        n_wl = len(wls)
        wl_vals = np.asarray(wls)

        # Output shape (before and after optional level summation)
        full_shape = (n_wl,) + st2_p.out_shape              # (n_wl, n_time, n_station, n_plev)
        if sum_levels:
            final_shape = (n_wl,) + tuple(
                n for d, n in zip(spatial_dims, st2_p.out_shape) if d != rh_vert_dim
            )
            final_dims: tuple = ("wavelength",) + tuple(d for d in spatial_dims if d != rh_vert_dim)
        else:
            final_shape = full_shape
            final_dims = ("wavelength",) + spatial_dims

        sp_coords = {d: layer_mass.coords[d].values for d in spatial_dims}
        if sum_levels:
            final_coords = {"wavelength": wl_vals, **{d: sp_coords[d] for d in spatial_dims if d != rh_vert_dim}}
        else:
            final_coords = {"wavelength": wl_vals, **sp_coords}

        aod_total_np = np.zeros(final_shape)
        abs_total_np = np.zeros(final_shape)
        aod_species_np: dict[str, np.ndarray] = {}
        abs_species_np: dict[str, np.ndarray] = {}

        for spec in species:
            opt_idx, is_philic = aeromap[spec]
            if is_philic:
                me_np = philic_mass_ext[spec]               # (n_wl, n_time, n_station, n_plev)
                ssa_np = philic_ssa[spec]
            else:
                opt = self._species_optics(spec, opt_idx, False, wns, wls)
                me_np = opt["mass_ext_hydrophobic"].values.reshape(n_wl, *(1,)*n_spatial)
                ssa_np = opt["ssa_hydrophobic"].values.reshape(n_wl, *(1,)*n_spatial)

            # Pressure-regrid mmr to ERA5 levels (Fortran, no xarray overhead)
            mmr_np = vinterp.interp_fld(
                fsrc=np.ascontiguousarray(
                    a[spec].values.transpose(_src_order).reshape(st2_p.src_stackshape)
                ),
                tgtlevs=p_tgtlevs_t,
                weights=p_weights_t,
            ).reshape(st2_p.out_shape)[np.newaxis]  # (1, n_time, n_station, n_plev)

            ext_np = mmr_np * lm_bc * me_np         # (n_wl, n_time, n_station, n_plev)
            absext_np = ext_np * (1.0 - ssa_np)
            if sum_levels:
                ext_np = ext_np.sum(axis=lev_axis_in_output)
                absext_np = absext_np.sum(axis=lev_axis_in_output)

            aod_total_np += ext_np
            abs_total_np += absext_np
            if per_species:
                aod_species_np[spec] = ext_np
                abs_species_np[spec] = absext_np

        ds = xr.Dataset()
        ds["AOD_total"] = xr.DataArray(aod_total_np, dims=final_dims, coords=final_coords).assign_attrs(units="1")
        ds["Abs_total"] = xr.DataArray(abs_total_np, dims=final_dims, coords=final_coords).assign_attrs(units="1")
        if per_species:
            for spec in species:
                ds[f"AOD_{spec}"] = xr.DataArray(aod_species_np[spec], dims=final_dims, coords=final_coords).assign_attrs(units="1")
                ds[f"Abs_{spec}"] = xr.DataArray(abs_species_np[spec], dims=final_dims, coords=final_coords).assign_attrs(units="1")
        return ds

_DS = TypeVar("_DS", xr.Dataset, xr.DataArray)


def angstrom_interpolation(aod_da: xr.DataArray, tgt_wl: float,
                           max_wl_dist: float = 50,
                           wl_dim: str = "wavelength_nm") -> xr.DataArray:
    """Interpolate AOD to a target wavelength using the Angstrom exponent.

    Uses the two wavelengths in ``aod_da`` that bracket ``tgt_wl`` to derive
    a local Angstrom exponent α, then evaluates τ = τ_lo * (λ_tgt/λ_lo)^(−α).
    """
    wl_values = aod_da[wl_dim].values.astype(float)
    wl_hi = float(wl_values[wl_values >= tgt_wl].min(initial=np.inf))
    wl_lo = float(wl_values[wl_values <= tgt_wl].max(initial=-np.inf))

    if wl_hi - tgt_wl > max_wl_dist or tgt_wl - wl_lo > max_wl_dist:
        raise ValueError(
            f"Target wavelength {tgt_wl} nm is too far from available wavelengths: {wl_values}"
        )

    if wl_hi == wl_lo:
        return aod_da.sel({wl_dim: wl_hi})

    aod_lo = aod_da.sel({wl_dim: wl_lo})
    aod_hi = aod_da.sel({wl_dim: wl_hi})

    ae = np.log(aod_lo / aod_hi) / np.log(wl_hi / wl_lo)  # α > 0 for typical AOD
    return aod_lo * (tgt_wl / wl_lo) ** (-ae)


def angstrom_fill_nans(
    aod_da: xr.DataArray,
    wl_tgt: float,
    max_wl_dist: float = 200.0,
    wl_dim: str = "wavelength_nm",
    verbose: bool = False
) -> None:
    """Fill NaN values at each wavelength using Angstrom interpolation from
    the nearest locally-valid bracketing wavelengths at each individual point.

    Operates **in-place** on ``aod_da``; no copy is made.

    The bracketing wavelengths are not fixed globally: for each (time, station)
    location the nearest available non-NaN wavelengths on both sides of the
    target are used.  All pairs of bracketing wavelengths within ``max_wl_dist``
    are tried from nearest to farthest; each pass fills only the points where
    both brackets are valid and the target is still NaN, so the closest valid
    pair always takes precedence.

    Original non-NaN values are always preserved.

    Parameters
    ----------
    aod_da
        AOD DataArray with a wavelength dimension (e.g. from
        ``AeronetCollection.get_obs_aod``).
    max_wl_dist
        Maximum allowed gap (nm) between the target and either bracketing
        wavelength.
    wl_dim
        Name of the wavelength coordinate.

    Returns
    -------
    xr.DataArray
        Same shape as input; NaN values filled where possible.
    """
    wl_sorted = np.sort(aod_da[wl_dim].values.astype(float))

    if not aod_da.sel({wl_dim: wl_tgt}).isnull().any():
        return

    # Candidates on each side, sorted nearest-first, filtered by max_wl_dist
    below = wl_sorted[(wl_sorted < wl_tgt) & ((wl_tgt - wl_sorted) <= max_wl_dist)][::-1]
    above = wl_sorted[(wl_sorted > wl_tgt) & ((wl_sorted - wl_tgt) <= max_wl_dist)]

    if len(below) == 0 or len(above) == 0:
        if verbose:
            print(f"No valid bracketing wavelengths found for {wl_tgt} nm within {max_wl_dist} nm")
        return

    for wl_lo in below:
        # if not aod_da.sel({wl_dim: wl_tgt}).isnull().any():
        #     if verbose:
        #         print(f"All NaN points at {wl_tgt} nm filled; stopping early")
        #     break
        for wl_hi in above:
            still_nan = aod_da.sel({wl_dim: wl_tgt}).isnull()
            if not still_nan.any():
                if verbose:
                    print(f"All NaN points at {wl_tgt} nm filled; stopping early")
                break

            aod_lo = aod_da.sel({wl_dim: wl_lo})
            aod_hi = aod_da.sel({wl_dim: wl_hi})

            # Fill only where target is still NaN AND both brackets are valid
            fill_mask = still_nan & aod_lo.notnull() & aod_hi.notnull()
            if not fill_mask.any():
                # if verbose:
                #     print(f"No valid points to fill at {wl_tgt} nm using {wl_lo} and {wl_hi} nm")
                continue

            if verbose:
                n_fill = int(fill_mask.sum().values)
                print(f"Filling {n_fill} NaN points at {wl_tgt} nm using {wl_lo} and {wl_hi} nm")

            ae = np.log(aod_lo / aod_hi) / np.log(wl_hi / wl_lo)
            fill = aod_lo * (wl_tgt / wl_lo) ** (-ae)
            aod_da.loc[{wl_dim: wl_tgt}] = xr.where(
                fill_mask, fill, aod_da.sel({wl_dim: wl_tgt})
            )

