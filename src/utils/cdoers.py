from typing import List

def gen_griddes_unstructured(lons : List[float],
                             lats : List[float]) -> str:
    """Generate griddes file for regridding using cdo"""
    if len(lons) != len(lats):
        raise ValueError("Longitude and latitude arrays must have the same length")
    npts = len(lons)
    griddes_lines = [
        "gridtype = unstructured",
        f"gridsize = {npts}",
        "xname = lon",
        "xdimname = col",
        "xlongname = longitude",
        "xunits = degrees_east",
        "yname = lat",
        "ydimname = col",
        "ylongname = latitude",
        "yunits = degrees_north"
    ]
    xvals = []
    yvals = []
    for i in range(npts):
        xvals.append(f"{lons[i]:.6f}")
        yvals.append(f"{lats[i]:.6f}")
    xvals = " ".join(xvals)
    yvals = " ".join(yvals)
    griddes_lines.append("xvals = " + xvals)
    griddes_lines.append("yvals = " + yvals)
    return "\n".join(griddes_lines)


def cdo_interpolate_2d(griddes_file: str, input_nc: str, output_nc: str):
    """
    Interpolate a 2D NetCDF file to a new grid using CDO.

    Parameters:
    - griddes_file: Path to the grid description file.
    - input_nc: Path to the input NetCDF file.
    - output_nc: Path where the interpolated NetCDF file will be saved.
    """
    import subprocess
    import shutil

    # Check that cdo is available
    if shutil.which("cdo") is None:
        raise RuntimeError("CDO command not found. CDO must be installed and available in PATH.")

    cmd_cdo_regrid = ["cdo", f"-remapbil,{griddes_file}", input_nc, output_nc]
    print(f"Running CDO command: {' '.join(cmd_cdo_regrid)}")
    subprocess.run(cmd_cdo_regrid, check=True)