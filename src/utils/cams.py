
from typing import Union

import numpy as np
import xarray as xr

from src.structs.collector import ObsCollection, StationsCollection

RENAMEDIC = {
    **{f"aermr{i:02d}" : f"Sea_Salt_bin{i}"
       for i in [1,2,3]},
    **{f"aermr{i+3:02d}" : f"Mineral_Dust_bin{i}"
       for i in [1,2,3]}
}

# -----------------------------------------------------------------------------
# Monthly climatology interpolation
# -----------------------------------------------------------------------------
def interpolate_monthly_clim(
    dset: xr.Dataset,
    dates: xr.DataArray,
) -> xr.Dataset:
    """Linearly interpolate a monthly climatology to arbitrary `dates`.

    The input `dset` **must** have a `month` coordinate/dimension.
    dates must have 'time' coordinate and dimension, but can also have other coordinates
    along the time dimension. If those coordinates are also a dimension in dset, then these are considered
    aligned to the time interpolation and the result will be contracted along those dimensions as well.

    For example, if dset has dimensions (month, lat) and dates has coordinates (time(time), lat(time)),
    then the output will be the interpolation along time of the corresponding lat,
    so will have only the time dimension.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        Monthly climatology with a `month` dimension.
    dates : xr.DataArray
        Target dates (datetime64).

    Returns
    -------
    xr.Dataset or xr.DataArray
        The same kind as input, interpolated to `dates` along a new/used `time` coordinate.
    """
    if len(dates.time) == 0:
        collapse_output = True
        dates = dates.isel(time=[0])
    else:
        collapse_output = False

    # Previous and following "anchor" mid-months around the target dates
    prev_month = (dates.values - np.timedelta64(14, "D")).astype("datetime64[M]") +\
          np.timedelta64(14, "D")
    foll_month = (prev_month + np.timedelta64(18, "D")).astype("datetime64[M]") +\
          np.timedelta64(14, "D")

    monthdelta = foll_month - prev_month
    thisdelta = dates - prev_month.astype("datetime64[ns]")
    timeweight_m = thisdelta / monthdelta  # in [0, 1]

    months_coords = {
        **{"time": dates},
        **{cname: coord
           for cname, coord in dates.coords.items()
           if cname != "month"},
    }

    intmonths_bot = xr.DataArray(
        data=prev_month.astype("datetime64[ns]"),
        dims=["time"],
        coords=months_coords
        ).dt.month
    intmonths_top = xr.DataArray(
        data=foll_month.astype("datetime64[ns]"),
        dims=["time"],
        coords=months_coords
        ).dt.month

    add_sel_kwargs = {str(cname): coord
                      for cname, coord in dates.coords.items()
                      if cname in dset.dims}

    # Take
    if "epoch" in dset.coords:
        epochs = dset.epoch.values
        max_epoch, min_epoch = epochs.max(), epochs.min()
        target_years = dates.dt.year.values
        lower_idx  = np.clip(np.searchsorted(epochs, target_years, side="right") - 1, 0, len(epochs)-1)
        upper_idx  = np.clip(np.searchsorted(epochs, target_years, side="left"), 0, len(epochs)-1)
        epoch_lower = epochs[lower_idx]
        epoch_upper = epochs[upper_idx]

        span = epoch_upper - epoch_lower
        timeweight_e = np.divide(
            target_years - epoch_lower,
            span,
            out=np.zeros_like(span, dtype=float),
            where=~np.isclose(span, 0.0, atol=1.e-2)
            )
        needed_epochs = np.unique(np.concat([epoch_lower, epoch_upper]))
        add_sel_kwargs["epoch"] = needed_epochs
    else:
        timeweight_e = []
        epoch_lower = []
        epoch_upper = []

    lower = dset.sel(month=intmonths_bot, **add_sel_kwargs).drop_vars("month")
    upper = dset.sel(month=intmonths_top, **add_sel_kwargs).drop_vars("month")

    dset_intp_m = (1 - timeweight_m) * lower + timeweight_m * upper
    if "epoch" in dset_intp_m.coords:
        timeweight_e = xr.DataArray(data=timeweight_e, dims=["time"])
        epoch_lower = xr.DataArray(data=epoch_lower, dims=["time"])
        epoch_upper = xr.DataArray(data=epoch_upper, dims=["time"])

        dset_intp = (1-timeweight_e) * dset_intp_m.sel(epoch=epoch_lower) +\
            timeweight_e * dset_intp_m.sel(epoch=epoch_upper)
    else:
        dset_intp = dset_intp_m

    if collapse_output:
        dset_intp = dset_intp.isel(time=0, drop=False)

    return dset_intp


def gen_cams_pointinterp(obs_coll : Union[ObsCollection, StationsCollection],
                         overwrite : bool = False) -> None:
    """Generate the point-interpolated CAMS climatology"""
    import os

    from .cdoers import cdo_interpolate_2d

    cams_clim_points_fpath = obs_coll.cams_clim_points_file

    if os.path.exists(cams_clim_points_fpath):
        if not overwrite:
            print(f"CAMS climatology interpolated to observation sites already exists at {os.path.basename(cams_clim_points_fpath)}. Nothing to do here.")
            return
        print(f"Overwriting existing file at {cams_clim_points_fpath}.")
        os.remove(cams_clim_points_fpath)


    griddes_file = obs_coll.griddes_file

    # Griddes file?
    if overwrite and os.path.exists(griddes_file):
        print(f"Overwriting existing griddes file at {os.path.basename(griddes_file)}.")
        os.remove(griddes_file)

    if not os.path.exists(griddes_file):
        print(f"Creating griddes file at {griddes_file}")
        from src.utils.cdoers import gen_griddes_unstructured
        obs_coll_locs = obs_coll.get_grid_xarray()
        griddes_str = gen_griddes_unstructured(obs_coll_locs.lon.values, obs_coll_locs.lat.values)
        with open(griddes_file, 'w') as f:
            f.write(griddes_str)
    else:
        print(f"Griddes file {os.path.basename(griddes_file)} exists.")

    cams_clim_fpath = obs_coll.cams_clim_file

    if not os.path.exists(cams_clim_fpath):
        raise FileNotFoundError(f"CAMS climatology file not found at expected location {cams_clim_fpath}.")
    ds_orig = xr.open_dataset(cams_clim_fpath)
    #stackdims = [d for d in ["epoch", "month"] if d in ds_orig.dims]
    #ds_orig = ds_orig.stack(time=stackdims).drop_vars(stackdims).transpose("time", ...).to_netcdf(cams_clim_restricted_fpath)

    # Hardcoded for now. If species with anthropogenic signal are needed,
    # Then this should handle also the epoch dimension
    from src.structs.collector import INPCollection
    if obs_coll.__class__ == INPCollection:
        spec_sel_sites = [f"{sp}_bin{i}" for sp in ["Mineral_Dust", "Sea_Salt"] for i in [1,2,3]] +\
              ["pressure"]
        ds_orig = ds_orig[spec_sel_sites]

    # All variables with the "epoch" dimension
    var_epoch = [v for v in ds_orig.data_vars if "epoch" in ds_orig[v].dims]
    var_nonepoch = [v for v in ds_orig.data_vars if "epoch" not in ds_orig[v].dims]


    cams_clim_restricted_points = []
    for i,var_group in enumerate([var_epoch, var_nonepoch]):
        if len(var_group) == 0:
            continue

        cams_clim_restricted_fpath = cams_clim_fpath.replace(".nc", f"_restricted{i}.nc")
        # Save to workdir subdirectory
        cams_clim_restricted_fpath = os.path.join(os.path.dirname(cams_clim_fpath), "workdir", os.path.basename(cams_clim_restricted_fpath))
        cams_clim_restricted_points_fpath = cams_clim_restricted_fpath.replace(".nc", "_points.nc")
        this_ds_group = ds_orig[var_group]
        has_epoch = "epoch" in this_ds_group.dims

        if "month" in this_ds_group.dims:
            if has_epoch:
                this_ds_group = this_ds_group.stack(time=["epoch", "month"])
                this_ds_group.reset_index("time").drop_vars(["epoch", "month"]).assign_coords(
                    time=xr.DataArray(np.arange(len(this_ds_group.time), dtype="float64"), dims="time", attrs={"units": "months since 1900-01-01"})
                ).transpose("time", ...).to_netcdf(cams_clim_restricted_fpath)
            else:
                this_ds_group.rename({"month": "time"}).to_netcdf(cams_clim_restricted_fpath)
        else:
            raise ValueError("CAMS climatology data must have a 'month' dimension for interpolation.")

        cdo_interpolate_2d(griddes_file, cams_clim_restricted_fpath, cams_clim_restricted_points_fpath)

        with xr.open_dataset(cams_clim_restricted_points_fpath, decode_times=False) as ds_sites:
            if has_epoch and "time" in ds_sites.dims:
                ds_sites = ds_sites.assign_coords(time=this_ds_group.indexes["time"]).unstack("time")
            elif "time" in ds_sites.dims:
                ds_sites = ds_sites.rename({"time": "month"})
                ds_sites = ds_sites.assign_coords(month=ds_orig["month"])
            cams_clim_restricted_points.append(ds_sites.load())
            os.remove(cams_clim_restricted_points_fpath)

    if len(cams_clim_restricted_points) == 0:
        raise ValueError(f"No interpolated cams clim datasets. Something went wrong with the interpolation.")

    xr.merge(cams_clim_restricted_points).to_netcdf(cams_clim_points_fpath)
    print(f"Created CAMS climatology interpolated to observation sites at {cams_clim_points_fpath}")

