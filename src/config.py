import os
import json

from pathlib import Path
from typing import Optional, Dict, Any, Literal

AeronetFreq = Literal["monthly", "daily"]
AeronetLevel = Literal["lev20", "lev15", "lev10"]

# Where this file is located
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

DATADIR = Path(os.path.join(SCRIPTDIR, "../data")).resolve()

CAMSCLIM_DATADIR = os.path.join(DATADIR, "cams_clim")
CAMSFREE_DATADIR = os.path.join(DATADIR, "cams_free")
INPDB_DATADIR = os.path.join(DATADIR, "inp_db")
AERONET_DATADIR= os.path.join(DATADIR, "aeronet")
ERA5_DATADIR = os.path.join(DATADIR, "era5")
VOLCAERO_DATADIR = os.path.join(DATADIR, "volc_aero")
MACV2SP_DATADIR = os.path.join(DATADIR, "macv2_sp")
MERRA2_DATADIR = os.path.join(DATADIR, "merra2")

CONFIGDICT_DEF: Dict[str, Any] = {
    "merra2_aeronet_file" : "", # "merra2.instM_2d_gas_Nx_AOD_MERGED_1degshifted_aeronet.nc",
    "macv2sp_aeronet_file" : "", #"synth_files/MACv2-SP_2Dfields_T63_1km_1980_2020_aeronet.nc",
    "volcaero_file" : "GLOSSAC_EVA_prescribed_1980_2020_aeronetwls.nc",
    "cams_free_sites_file" : "*_sites_zarr",
    "cams_free_aeronet_file" : "*_aeronet_zarr",
    "cams_clim_file" : "aerosol_cams_climatology_49r2_1951-2019_4D.nc",
    "inpdb_csvfile" : "20260417_Immersion_2003_2020_INPDB.csv",
    "aeronet_dir" : "aod_monthly_20260624",
    "lonlat_approx" : 0.1,
    "t_approx_h" : True,
    "aeronet_freq" : "monthly",
    "aeronet_lev" : "lev20",
}

CONFIGDICT : Dict[str, Any] = {}

def get_merra2_aeronet_path() -> str:
    """Get the path to the MERRA-2 aeronet file based on the current configuration."""
    if "merra2_aeronet_file" not in CONFIGDICT:
        raise ValueError("merra2_aeronet_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(MERRA2_DATADIR, CONFIGDICT["merra2_aeronet_file"])

def get_macv2sp_aeronet_path() -> str:
    """Get the path to the MACv2-SP aeronet file based on the current configuration."""
    if "macv2sp_aeronet_file" not in CONFIGDICT:
        raise ValueError("macv2sp_aeronet_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(MACV2SP_DATADIR, CONFIGDICT["macv2sp_aeronet_file"])

def get_volcaero_path() -> str:
    """Get the path to the VolcAero file based on the current configuration."""
    if "volcaero_file" not in CONFIGDICT:
        raise ValueError("volcaero_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(VOLCAERO_DATADIR, CONFIGDICT["volcaero_file"])

def get_era5_aeronet_path(geopot : bool = False, aeronet_freq : AeronetFreq = "monthly") -> str:
    """Get the path to the ERA5 aeronet file based on the current configuration."""
    if "aeronet_freq" not in CONFIGDICT:
        raise ValueError("aeronet_freq not set in CONFIGDICT. Call digest_config first.")
    if geopot:
        return os.path.join(ERA5_DATADIR, f"synth_files/era5_z_aeronet.nc")
    return os.path.join(ERA5_DATADIR, f"synth_files/era5_1980to2020_{aeronet_freq}_aeronet_zarr")

def get_cams_free_sites_path() -> str:
    """Get the path to the CAMS free file based on the current configuration."""
    if "cams_free_sites_file" not in CONFIGDICT:
        raise ValueError("cams_free_sites_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSFREE_DATADIR, CONFIGDICT["cams_free_sites_file"])

def get_cams_free_aeronet_path() -> str:
    """Get the path to the CAMS free aeronet file based on the current configuration."""
    if "cams_free_aeronet_file" not in CONFIGDICT:
        raise ValueError("cams_free_aeronet_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSFREE_DATADIR, CONFIGDICT["cams_free_aeronet_file"])

def get_cams_clim_path() -> str:
    """Get the path to the CAMS climatology file based on the current configuration."""
    if "cams_clim_file" not in CONFIGDICT:
        raise ValueError("cams_clim_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSCLIM_DATADIR, CONFIGDICT["cams_clim_file"])

def get_cams_clim_sites_path() -> str:
    """Path to CAMS climatology interpolated to INDB sites"""
    clim_path = get_cams_clim_path()
    return clim_path.replace(".nc", "_sites.nc")

def get_cams_clim_aeronet_path() -> str:
    """Path to CAMS climatology interpolated to AERONET sites"""
    clim_path = get_cams_clim_path()
    return clim_path.replace(".nc", "_aeronet.nc")

def get_aeronet_dir_path() -> str:
    """Get the path to the directory containing aeronet sites"""
    if "aeronet_dir" not in CONFIGDICT:
        raise ValueError("aeronet_dir not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(AERONET_DATADIR, CONFIGDICT["aeronet_dir"])

def get_inpdb_csv_path() -> str:
    """Get the path to the INPDB CSV file based on the current configuration."""
    if "inpdb_csvfile" not in CONFIGDICT:
        raise ValueError("inpdb_csvfile not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(INPDB_DATADIR, CONFIGDICT["inpdb_csvfile"])

def clear_config() -> None:
    """
    Clears configuration. Further usage needs a new call to digest_config.
    """
    CONFIGDICT = {}

def digest_config(config_path: Optional[str] = None) -> None:
    """
    Load and validate configuration from a JSON file and merge into CONFIGDICT.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Raises
    ------
    ValueError
        If a config key in the file is not present in the default CONFIGDICT.
    """
    # Read configurations
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            cfg_in: Dict[str, Any] = json.load(config_file)
    else:
        cfg_in = {}

    # Validate keys: anything not in defaults (and not prefixed with "other_") is an error
    for key in cfg_in:
        if not key.startswith("other_") and key not in CONFIGDICT_DEF:
            raise ValueError(
                f"config key {key} unknown. Verify spelling errors."
            )

    # Merge (excluding "other_*" keys which are ignored by design)
    # and log CONFIGDICT population
    for key, val in CONFIGDICT_DEF.items():
        if key.startswith("other_"):
            continue
        if key in cfg_in:
            CONFIGDICT[key] = cfg_in[key]
            print(f"Set {key:>20} to {CONFIGDICT[key]}")
        else:
            print(f"Using default value for {key}: {val}")
            CONFIGDICT[key] = val


    print(f"Successfully read configuration from {config_path}")