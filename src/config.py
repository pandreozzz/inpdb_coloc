import os
import json

from pathlib import Path
from typing import Optional, Dict, Any

# Where this file is located
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

DATADIR = Path(os.path.join(SCRIPTDIR, "../data")).resolve()

CAMSCLIM_DATADIR = os.path.join(DATADIR, "cams_clim")
CAMSFREE_DATADIR = os.path.join(DATADIR, "cams_free")
INPDB_DATADIR = os.path.join(DATADIR, "inp_db")


CONFIGDICT_DEF: Dict[str, Any] = {
    "cams_free_file" : "*_zarr",
    "cams_clim_file" : "aerosol_cams_climatology_49r2_1951-2019_4D.nc", 
    "inpdb_csvfile" : "20260417_Immersion_2003_2020_INPDB.csv",
    "lonlat_approx" : 0.1,
    "t_approx_h" : True
}

CONFIGDICT : Dict[str, Any] = {}

def get_cams_free_path() -> str:
    """Get the path to the CAMS free file based on the current configuration."""
    if "cams_free_file" not in CONFIGDICT:
        raise ValueError("cams_free_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSFREE_DATADIR, CONFIGDICT["cams_free_file"])

def get_cams_clim_path() -> str:
    """Get the path to the CAMS climatology file based on the current configuration."""
    if "cams_clim_file" not in CONFIGDICT:
        raise ValueError("cams_clim_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSCLIM_DATADIR, CONFIGDICT["cams_clim_file"])

def get_cams_clim_sites_path() -> str:
    """Path to CAMS climatology interpolated to INDB sites"""
    clim_path = get_cams_clim_path()
    return clim_path.replace(".nc", "_sites.nc")

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