"""Handle MACv2-SP files"""

from typing import Union, List, Dict, Any

from src.structs.collector import ObsCollection, StationsCollection

import numpy as np

import xarray as xr

def gen_macv2sp_pointinterp(macv2sp_points_fpath : str,
                            obs_coll : Union[ObsCollection, StationsCollection],
                            overwrite : bool = False) -> None:
    """Generate the point-interpolated MACv2-SP data"""
    import os
    from src.config import MACV2SP_DATADIR

    if os.path.exists(macv2sp_points_fpath):
        if not overwrite:
            print(f"MACv2-SP data interpolated to observation sites already exists at {os.path.basename(macv2sp_points_fpath)}. Nothing to do here.")
            return
        print(f"Overwriting existing file at {macv2sp_points_fpath}.")
        os.remove(macv2sp_points_fpath)

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
    grp_match = re.match(r"(.+)_(\d+)_(\d+)_(?:[^_]*)_(?:[^_]*)\.nc", os.path.basename(macv2sp_points_fpath))
    assert grp_match is not None and len(grp_match.groups()) == 3, f"Unexpected MACv2-SP filename format: {macv2sp_points_fpath}"
    macv2sp_file_prefix = grp_match.groups()[0]
    assert macv2sp_file_prefix != "", f"Unexpected MACv2-SP filename format: {macv2sp_points_fpath}"
    ini_year = int(grp_match.groups()[1])
    end_year = int(grp_match.groups()[2])

    orig_files_pattern = os.path.join(MACV2SP_DATADIR, f"{macv2sp_file_prefix}_{ini_year}_{end_year}_*nm.nc")

    from glob import glob
    orig_files = glob(orig_files_pattern)
    if orig_files == []:
        raise ValueError(f"No MACv2-SP files found for the given pattern: {orig_files_pattern}")

    import tempfile
    workdir = tempfile.TemporaryDirectory(dir=MACV2SP_DATADIR, prefix="workdir")
    dest_files = []
    for orig_file in orig_files:
        print(f"Interpolating MACv2-SP file {os.path.basename(orig_file)} to observation sites.")
        from src.utils.cdoers import cdo_interpolate_2d
        dest_file = os.path.join(workdir.name, os.path.basename(orig_file).replace(macv2sp_file_prefix, f"{macv2sp_file_prefix}_points"))
        cdo_interpolate_2d(griddes_file, orig_file, dest_file)
        dest_files.append(dest_file)

    open_macv2sp_output(dest_files).to_netcdf(macv2sp_points_fpath)

    # Remove the temporary files
    for dest_file in dest_files:
        os.remove(dest_file)


def open_macv2sp_output(output_path : Union[str, List[str]]) -> xr.Dataset:
    """Load data from Macv2-SP output file(s). output_path can also have wildcards for multiple locations
    Assumes one file per wavlength and the wavelength indicated as <filename_XXXnm.nc>
    """
    import re

    def _process_times(ds):
        import re
        grp_match = re.search(r"(.+) since (.+)", ds.time.units)
        assert grp_match is not None and len(grp_match.groups()) == 2, f"Unexpected time units format: {ds.time.units}"
        since_time_split = grp_match.groups()[1].split(" ")
        y, m, d = since_time_split[0].split("-")
        htime = since_time_split[1] if len(since_time_split) > 1 else "00:00:00"
        ini_time = np.datetime64(f"{int(y):4d}-{int(m):02d}-{int(d):02d}T{htime}")
        step_unit = grp_match.groups()[0].lower().strip()
        if step_unit == "months":
            times = ini_time + ds.time.values*365.25/12*np.timedelta64(1, "D")
            # Move day to 15
            times = (times.astype("datetime64[M]") + np.timedelta64(14, "D")).astype("datetime64[ns]")
        elif step_unit == "days":
            times = ini_time + ds.time.values*np.timedelta64(1, "D")
        else:
            raise ValueError(f"Time {step_unit} not recognised!")
        ds["time"] = times
        return ds

    from glob import glob
    if not isinstance(output_path, list):
        output_path = [output_path]
    all_files = []
    for path in output_path:
        all_files.extend(glob(path))
    all_files = list(set(all_files))

    if all_files == []:
        raise ValueError("No files found for the given output_path(s).")

    opends_kwargs : Dict[str, Any]= {"decode_times" : False} #{"combine": "nested", "concat_dim": "time"}
    # Are they zarr archives?
    if all_files[0].endswith("zarr"):
        print("Opening archives as zarr")
        opends_kwargs["engine"] = "zarr"
    else:
        print("Opening files as netCDF")
        opends_kwargs["engine"] = "netcdf4"


    # Concat by wavelength
    this_data = xr.concat(
        [_process_times(xr.open_dataset(f, **opends_kwargs)).expand_dims(
            wavelength=[int(re.search(r".*_(\d+)nm['_zarr','.nc']{1}", f).group(1))] # type: ignore
            )
            for f in all_files],
        dim="wavelength"
    )

    wl_attrs = {"units" : "nm"}
    for k, v in wl_attrs.items():
        this_data["wavelength"].attrs[k] = v

    return this_data
