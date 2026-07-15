"""MERRA2 utils"""

from typing import Union

from src.structs.collector import ObsCollection, StationsCollection


def gen_merra2_pointinterp(merra2_points_fpath : str,
                           obs_coll : Union[ObsCollection, StationsCollection],
                           overwrite : bool = False) -> None:
    """Generate the point-interpolated MERRA2 data"""
    import os
    import numpy as np
    import xarray as xr

    from src.config import MERRA2_DATADIR

    if os.path.exists(merra2_points_fpath):
        if not overwrite:
            print(f"MERRA2 data interpolated to observation sites already exists at {os.path.basename(merra2_points_fpath)}. Nothing to do here.")
            return
        print(f"Overwriting existing file at {merra2_points_fpath}.")
        os.remove(merra2_points_fpath)

    griddes_file = obs_coll.griddes_file

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


    import re
    grp_match = re.match(r"(.+_MERGED_[^_]+)_(?:[^_]+)_(?:[^_]+)\.nc",
                          os.path.basename(merra2_points_fpath))
    if grp_match is None or len(grp_match.groups()) != 1:
        raise ValueError(f"Unexpected MERRA2 filename format: {merra2_points_fpath}. The expected format is <prefix>_MERGED_<gridspecs>_<points>.nc")
    merra2_file_prefix = grp_match.groups()[0]

    assert merra2_file_prefix != "", f"Unexpected MERRA2 filename format: {merra2_points_fpath}"

    orig_files_pattern = os.path.join(MERRA2_DATADIR, f"{merra2_file_prefix}.nc")

    from glob import glob
    if not os.path.exists(orig_files_pattern):
        raise FileNotFoundError(f"No MERRA2 files found matching pattern: {orig_files_pattern}")

    from src.utils.cdoers import cdo_interpolate_2d
    merra2_points_fpath_tmp = merra2_points_fpath + ".tmp"
    cdo_interpolate_2d(griddes_file, orig_files_pattern, merra2_points_fpath_tmp)
    ds_points_tmp = xr.open_dataset(merra2_points_fpath_tmp)
    # Move day to 15 of the month
    ds_points_tmp["time"] = [
        np.datetime64(f"{str(t)[:7]}-15") for t in ds_points_tmp.time.values
    ]
    ds_points_tmp.to_netcdf(merra2_points_fpath)
    os.remove(merra2_points_fpath_tmp)


