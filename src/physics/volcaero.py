"""Handle volcanic aerosols from GLOSSAC (for now)"""
import os
from functools import lru_cache

from ..config import get_volcaero_path
import xarray as xr
import numpy as np

@lru_cache(maxsize=4)
def _get_glossac_aod_cached(glossac_aero_path: str) -> xr.DataArray:
    """Load and preprocess GloSSAC once per file path."""
    if not os.path.exists(glossac_aero_path):
        raise FileNotFoundError(f"GLOSSAC file not found: {glossac_aero_path}")

    # Load eagerly to release the file handle and keep a reusable in-memory array.
    with xr.open_dataset(glossac_aero_path) as ds:
        ds = ds.load()

    ds["time"] = [
        np.datetime64(f"{int(t/100)}-{round((t/100%1)*100):02d}-15")
        for t in ds.time.values
    ]
    ds = ds.rename({"wavelengths_glossac": "wavelength_nm"})
    ds = xr.concat(
        [
            ds.sel(lat=[90], method="nearest").assign_coords(lat=[90]),
            ds,
            ds.sel(lat=[-90], method="nearest").assign_coords(lat=[-90]),
        ],
        dim="lat",
    ).sortby("lat", ascending=False)
    return ds["Glossac_Aerosol_Optical_Depth"].isel(alt=-1)


def get_glossac_aod(glossac_aero_path: str | None = None) -> xr.DataArray:
    """Get the GLOSSAC volcanic aerosol dataset."""
    if glossac_aero_path is None:
        glossac_aero_path = get_volcaero_path()
    return _get_glossac_aod_cached(os.path.abspath(glossac_aero_path))