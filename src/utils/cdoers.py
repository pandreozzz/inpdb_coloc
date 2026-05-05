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