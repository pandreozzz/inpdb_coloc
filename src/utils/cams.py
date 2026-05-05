
from typing import Union

import numpy as np
import xarray as xr

from src.structs.collector import INPCollection

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
    dset: Union[xr.Dataset, xr.DataArray],
    dates: xr.DataArray,
) -> Union[xr.Dataset, xr.DataArray]:
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


def gen_cams_pointinterp(inp_coll : INPCollection, overwrite : bool = False) -> None:
    """Generate the point-interpolated CAMS climatology"""
    import os
    from src.config import get_cams_clim_path, get_cams_clim_sites_path, get_inpdb_csv_path, CONFIGDICT

    cams_clim_sites_fpath = get_cams_clim_sites_path()
    
    if os.path.exists(cams_clim_sites_fpath):
        if overwrite:
            print(f"CAMS climatology interpolated to INPDB sites already exists at {os.path.basename(cams_clim_sites_fpath)}. Nothing to do here.")
            return
        print(f"Overwriting existing file at {cams_clim_sites_fpath}.")
        os.remove(cams_clim_sites_fpath)


    griddes_file = get_inpdb_csv_path().replace(".csv", "_griddes.txt")

    inp_coll_locs = inp_coll.sparse.get_grid_xarray()

    # Griddes file?
    if overwrite and os.path.exists(griddes_file):
            print(f"Overwriting existing griddes file at {os.path.basename(griddes_file)}.")
            os.remove(griddes_file)

    if not os.path.exists(griddes_file):
        print(f"Creating griddes file at {griddes_file}")
        from src.utils.cdoers import gen_griddes_unstructured
        griddes_str = gen_griddes_unstructured(inp_coll_locs.lon.values, inp_coll_locs.lat.values)
        with open(griddes_file, 'w') as f:
            f.write(griddes_str)
    else:
        print(f"Griddes file {os.path.basename(griddes_file)} exists.")

    cams_clim_fpath = get_cams_clim_path()
    cams_clim_restricted_fpath = cams_clim_fpath.replace(".nc", "_restricted.nc")


    # Interpolation with cdo
    import shutil
    import subprocess

    #Check that cdo is available
    if shutil.which("cdo") is None:
        raise RuntimeError("CDO command not found. CDO must be installed and available in PATH.")
    else:
        print("CDO command found.")

    ds_orig = xr.open_dataset(cams_clim_fpath)
    #stackdims = [d for d in ["epoch", "month"] if d in ds_orig.dims]
    #ds_orig = ds_orig.stack(time=stackdims).drop_vars(stackdims).transpose("time", ...).to_netcdf(cams_clim_restricted_fpath)

    # Hardcoded for now. If species with anthropogenic signal are needed, 
    # Then this should handle also the epoch dimension
    spec_sel = [f"{sp}_bin{i}" for sp in ["Mineral_Dust", "Sea_Salt"] for i in [1,2,3]]
    ds_orig[spec_sel].rename(month="time").drop_vars("time").to_netcdf(cams_clim_restricted_fpath)

    cmd_cdo_regrid = ["cdo", f"-remapbil,{griddes_file}", cams_clim_restricted_fpath, cams_clim_sites_fpath]

    print(f"Running CDO command: {' '.join(cmd_cdo_regrid)}")
    subprocess.run(cmd_cdo_regrid, check=True)

    print(f"Removing CAMS {os.path.basename(cams_clim_restricted_fpath)}")
    os.remove(cams_clim_restricted_fpath)

    # Reassign the multiindex and unstack
    with xr.open_dataset(cams_clim_sites_fpath) as ds_sites:
        cams_clim_sites = ds_sites.rename(time="month").assign_coords(month=ds_orig["month"]).load()

    ds_orig.close()
    cams_clim_sites.to_netcdf(cams_clim_sites_fpath)

    


