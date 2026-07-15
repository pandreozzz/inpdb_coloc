import os
import json

from pathlib import Path
from typing import Optional, Dict, Any, Literal

AeronetDataProduct = Literal["AOD", "INV"]
AeronetFreq = Literal["monthly", "daily"]
AeronetLevel = Literal["lev20", "lev15", "lev10"]

ERA5Frequency = Literal["monthly", "daily"]

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
    "merra2_aeronet_filelike" : "merra2.instM_2d_gas_Nx_AOD_MERGED_1degshifted_aeronet.nc",
    "macv2sp_aeronet_filelike" : "MACv2-SP_2Dfields_T63_1km_1980_2020_aeronet.nc",
    "volcaero_file" : "GLOSSAC_EVA_prescribed_1980_2020_aeronetwls.nc",
    "lonlat_approx" : 0.1,
    "t_approx_h" : True,
    "cams_clim_file" : "aerosol_cams_climatology_49r2_1951-2019_4D.nc",
    "inpdb_csvfile" : "INPDB_2003-2020_immersion_20260708_124249_IFS_colocation.csv",
    "inpdb_tag" : "20260708",
    "aeronet_freq" : "monthly",
    "aeronet_tag" : "20260701",
    "aeronet_lev" : "lev20",
    "era5_frequency" : "monthly"
}

CONFIGDICT : Dict[str, Any] = {}



def get_merra2_aeronet_path() -> str:
    """Get the path to the MERRA-2 aeronet file based on the current configuration."""
    if "merra2_aeronet_filelike" not in CONFIGDICT:
        raise ValueError("merra2_aeronet_filelike not set in CONFIGDICT. Call digest_config first.")
    if CONFIGDICT["merra2_aeronet_filelike"] is None:
        return ""
    this_tag = f"aeronet-{CONFIGDICT['aeronet_freq']}_AOD-{CONFIGDICT['aeronet_tag']}"
    return os.path.join(MERRA2_DATADIR, "point_files",
                        CONFIGDICT["merra2_aeronet_filelike"].replace("aeronet.nc", f"{this_tag}.nc"))

def get_macv2sp_aeronet_path() -> str:
    """Get the path to the MACv2-SP aeronet file based on the current configuration."""
    if "macv2sp_aeronet_filelike" not in CONFIGDICT:
        raise ValueError("macv2sp_aeronet_filelike not set in CONFIGDICT. Call digest_config first.")
    if CONFIGDICT["macv2sp_aeronet_filelike"] is None:
        return ""
    this_tag = f"aeronet-{CONFIGDICT['aeronet_freq']}_AOD-{CONFIGDICT['aeronet_tag']}"
    return os.path.join(MACV2SP_DATADIR, "point_files",
                        CONFIGDICT["macv2sp_aeronet_filelike"].replace("aeronet.nc", f"{this_tag}.nc"))

def get_volcaero_path() -> str:
    """Get the path to the VolcAero file based on the current configuration."""
    if "volcaero_file" not in CONFIGDICT:
        raise ValueError("volcaero_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(VOLCAERO_DATADIR, CONFIGDICT["volcaero_file"])

def get_era5_path(ini_year : int, end_year : int,
                  geopot : bool = False,
                  era5_frequency : Optional[ERA5Frequency] = None,
                  data_tag : str = ""
                  ) -> str:
    """Get the path to the ERA5 file based on the current configuration."""
    if era5_frequency is None:
        if "era5_frequency" not in CONFIGDICT:
            raise ValueError("era5_frequency not set in CONFIGDICT. Call digest_config first.")
        era5_frequency = CONFIGDICT["era5_frequency"]
    if geopot:
        return os.path.join(ERA5_DATADIR, f"era5_z_{data_tag}.nc")
    return os.path.join(ERA5_DATADIR, f"synth_files/era5_{ini_year}to{end_year}_{era5_frequency}_{data_tag}_zarr")

def get_era5_aeronet_path(ini_year : int = 1980, end_year : int = 2020,
                          aeronet_tag : Optional[str] = None,
                          geopot : bool = False,
                          era5_frequency : Optional[ERA5Frequency] = None,
                          aeronet_freq : AeronetFreq = "monthly",
                          aeronet_product : AeronetDataProduct = "AOD"
                          ) -> str:
    """Get the path to the ERA5 aeronet file based on the current configuration."""
    if aeronet_tag is None:
        if "aeronet_tag" not in CONFIGDICT:
            raise ValueError("aeronet_tag not set in CONFIGDICT. Call digest_config first.")
        aeronet_tag = CONFIGDICT["aeronet_tag"]
    this_tag = f"aeronet-{aeronet_freq}_{aeronet_product}-{aeronet_tag}"
    return get_era5_path(ini_year=ini_year, end_year=end_year, geopot=geopot, era5_frequency=era5_frequency, data_tag=this_tag)

def get_era5_inpdb_path(inpdb_tag : Optional[str] = None,
                        geopot : bool = False,
                        era5_frequency : Optional[ERA5Frequency] = None
                        ) -> str:
    """Get the path to the ERA5 inpdb file based on the current configuration."""
    if inpdb_tag is None:
        if "inpdb_tag" not in CONFIGDICT:
            raise ValueError("inpdb_tag not set in CONFIGDICT. Call digest_config first.")
        inpdb_tag = CONFIGDICT["inpdb_tag"]
    this_tag = f"inpdb_{inpdb_tag}"
    return get_era5_path(ini_year=2003, end_year=2020, geopot=geopot, era5_frequency=era5_frequency, data_tag=this_tag)

def get_cams_free_sites_path() -> str:
    """Get the path to the CAMS free file based on the current configuration."""
    this_tag = f"inpdb_{CONFIGDICT['inpdb_tag']}"
    return os.path.join(CAMSFREE_DATADIR, f"*_{this_tag}_zarr")

def get_cams_free_aeronet_path(data_product : AeronetDataProduct) -> str:
    """Get the path to the CAMS free aeronet file based on the current configuration."""
    this_tag = f"_aeronet-{CONFIGDICT['aeronet_freq']}_{data_product}-{CONFIGDICT['aeronet_tag']}"
    return os.path.join(CAMSFREE_DATADIR,
                        f"*{this_tag}_zarr")

def get_cams_clim_path() -> str:
    """Get the path to the CAMS climatology file based on the current configuration."""
    if "cams_clim_file" not in CONFIGDICT:
        raise ValueError("cams_clim_file not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(CAMSCLIM_DATADIR, CONFIGDICT["cams_clim_file"])

def get_cams_clim_sites_path() -> str:
    """Path to CAMS climatology interpolated to INDB sites"""
    clim_path = get_cams_clim_path()
    this_tag = f"inpdb_{CONFIGDICT['inpdb_tag']}"
    return os.path.join(CAMSCLIM_DATADIR, "point_files",
                        os.path.basename(clim_path).replace(".nc", f"_{this_tag}.nc"))

def get_cams_clim_aeronet_path(data_product : AeronetDataProduct) -> str:
    """Path to CAMS climatology interpolated to AERONET sites"""
    clim_path = get_cams_clim_path()
    this_tag = f"aeronet-{CONFIGDICT['aeronet_freq']}_{data_product}-{CONFIGDICT['aeronet_tag']}"
    return os.path.join(CAMSCLIM_DATADIR, "point_files",
                        os.path.basename(clim_path).replace(".nc", f"_{this_tag}.nc"))

def get_aeronet_dir_path(data_product : AeronetDataProduct = "AOD") -> str:
    """Get the path to the directory containing aeronet sites"""
    if "aeronet_freq" not in CONFIGDICT:
        raise ValueError("aeronet_freq not set in CONFIGDICT. Call digest_config first.")
    if "aeronet_lev" not in CONFIGDICT:
        raise ValueError("aeronet_lev not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(AERONET_DATADIR,
                        CONFIGDICT["aeronet_freq"],
                        data_product,
                        CONFIGDICT["aeronet_tag"],
                        CONFIGDICT["aeronet_lev"]
                        )

def get_inpdb_csv_path() -> str:
    """Get the path to the INPDB CSV file based on the current configuration."""
    if "inpdb_tag" not in CONFIGDICT:
        raise ValueError("inpdb_tag not set in CONFIGDICT. Call digest_config first.")
    if "inpdb_csvfile" not in CONFIGDICT:
        raise ValueError("inpdb_csvfile not set in CONFIGDICT. Call digest_config first.")
    return os.path.join(INPDB_DATADIR, CONFIGDICT["inpdb_tag"], CONFIGDICT["inpdb_csvfile"])

def clear_config() -> None:
    """
    Clears configuration. Further usage needs a new call to digest_config.
    """
    global CONFIGDICT
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