from typing import Union, Optional, List, Tuple

from src.config import get_era5_aeronet_path, get_macv2sp_aeronet_path, get_merra2_aeronet_path
from src.structs.collector import INPCollection, AeronetCollection

def build_inp_collection(inpdb_csv_path : Optional[str] = None,
                         cams_free_sites_path : Union[None, str, List[str]] = None,
                         cams_clim_sites_path : Optional[str] = None,
                         lonlat_approx : Optional[float] = None,
                         lon_to_degeast : Optional[bool] = None,
                         t_approx_h : Optional[bool] = None,
                         sortbytime : bool = False,
                         timerange: Optional[Tuple[str, str]] = None,
                         n_lines : Optional[int] = None
                         ) -> INPCollection:
    """Helper function to build collection from file path and parameters
    n_lines : how many lines of the csv to read (useful for testing)
    pass path "" to ignore: e.g. cams_free_sites_path = "" will skip loading CAMS free data
    """
    import os
    from glob import glob
    from src.config import get_inpdb_csv_path, get_cams_free_sites_path, \
        get_cams_clim_sites_path, CONFIGDICT


    if CONFIGDICT == {}:
        from src.config import digest_config
        print("Warning: CONFIGDICT is empty. Calling digest_config with default parameters.")
        digest_config()

    if inpdb_csv_path is None:
        inpdb_csv_path = get_inpdb_csv_path()
    # Preliminary sanity checks
    elif not os.path.exists(inpdb_csv_path):
        raise ValueError(f"Provided path {inpdb_csv_path} not found.")

    if cams_clim_sites_path is None:
        cams_clim_sites_path = get_cams_clim_sites_path()
    # Preliminary sanity checks
    elif not os.path.exists(cams_clim_sites_path) and not cams_clim_sites_path == "":
        raise ValueError(f"Provided path {cams_clim_sites_path} not found.")

    if cams_free_sites_path is None:
        cams_free_sites_path = get_cams_free_sites_path()
    # Preliminary sanity checks
    elif isinstance(cams_free_sites_path, str) and not cams_free_sites_path == "":
        cams_free_sites_path = [cams_free_sites_path]
        allfiles = []
        for path in cams_free_sites_path:
            allfiles.extend(glob(path))
        if allfiles == []:
            raise ValueError(f"No files found for the given cams_free_sites_path(s).")

    # DEFAULTS vs overridden values
    if lonlat_approx is None:
        lonlat_approx = CONFIGDICT.get("lonlat_approx", None)
    if lon_to_degeast is None:
        lon_to_degeast = CONFIGDICT.get("lon_to_degeast", True)
    if t_approx_h is None:
        t_approx_h = CONFIGDICT.get("t_approx_h", True)

    if not os.path.exists(cams_clim_sites_path):
        from src.utils.cams import gen_cams_pointinterp
        print(f"CAMS climatology interpolated to INPDB sites not found at {cams_clim_sites_path}. Will be generated")
        this_cams_clim_sites_path = None
    else:
        this_cams_clim_sites_path = cams_clim_sites_path

    assert isinstance(lon_to_degeast, bool)
    assert isinstance(t_approx_h, bool)

    inp_coll = INPCollection(inpdb_path=inpdb_csv_path,
                             cams_clim_path=this_cams_clim_sites_path,
                             cams_free_path=cams_free_sites_path,
                             n_lines=n_lines,
                             lonlat_approx=lonlat_approx,
                             lon_to_degeast=lon_to_degeast,
                             t_approx_h=t_approx_h, sortbytime=sortbytime,
                             timerange=timerange)

    if this_cams_clim_sites_path is None:
        from src.utils.cams import gen_cams_pointinterp
        gen_cams_pointinterp(inp_coll)
        inp_coll.load_cams_clim(get_cams_clim_sites_path())

    return inp_coll

def build_aeronet_collection(
    aeronet_sites_dir : Optional[str] = None,
    cams_free_aeronet_path : Union[None, str, List[str]] = None,
    cams_clim_aeronet_path : Optional[str] = None,
    macv2sp_aeronet_path : Optional[str] = None,
    macv2nat_aeronet_path : Optional[str] = None,
    merra2_aeronet_path : Optional[str] = None,
    lon_to_degeast : Optional[bool] = None,
    t_approx_h : Optional[bool] = None,
    timerange: Optional[Tuple[str, str]] = None,
    overwrite : bool = False
    ) -> AeronetCollection:
    """Build collection from file path and parameter

    Parameters
    ----------
    aeronet_sites_dir : str, optional
        Path to the directory containing Aeronet site data.
    Custom model datasets for Aeronet sites (point interpolated). If not provided,
    they will follow the main config (src/config.py) settings. To
    skip loading any of these, just pass an empty string "".
    cams_free_aeronet_path : str or list of str, optional (default: None)
        Path(s) to the CAMS free output files for Aeronet sites.
    cams_clim_aeronet_path : str, optional (default: None)
        Path to the CAMS climatology output file for Aeronet sites.
    macv2sp_aeronet_path : str, optional (default: None)
        Path to the MACv2-SP output file for Aeronet sites.
    merra2_aeronet_path : str, optional (default: None)
        Path to the MERRA-2 output file for Aeronet sites.
    lon_to_degeast : bool, optional (default: None)
        Whether to convert longitudes to degrees east. If None, will use the default from config.
    t_approx_h : bool, optional (default: None)
        Whether to approximate time to the nearest hour. If None, will use the default from config.
    timerange : tuple of str, optional (default: None)
        Time range for filtering the data, in the format (start_time, end_time).
    overwrite : bool (default: False)
        Whether to overwrite existing files when generating the point interpolated data
        (cams_clim or macv2sp or merra2).
    """

    import os
    from glob import glob
    from src.config import get_aeronet_dir_path, get_cams_free_aeronet_path, \
        get_cams_clim_aeronet_path, get_macv2sp_aeronet_path, get_macv2nat_aeronet_path, CONFIGDICT

    if CONFIGDICT == {}:
        from src.config import digest_config
        print("Warning: CONFIGDICT is empty. Calling digest_config with default parameters.")
        digest_config()

    if aeronet_sites_dir is None:
        aeronet_sites_dir = get_aeronet_dir_path()
    elif not os.path.exists(aeronet_sites_dir):
        raise ValueError(f"Provided path {aeronet_sites_dir} not found.")

    if cams_clim_aeronet_path is None:
        cams_clim_aeronet_path = get_cams_clim_aeronet_path(data_product="AOD")

    if macv2sp_aeronet_path is None:
        macv2sp_aeronet_path = get_macv2sp_aeronet_path()
    # Preliminary sanity checks
    elif macv2sp_aeronet_path != "" and not os.path.exists(macv2sp_aeronet_path):
        raise ValueError(f"Provided path {macv2sp_aeronet_path} not found.")

    if macv2nat_aeronet_path is None:
        macv2nat_aeronet_path = get_macv2nat_aeronet_path()
    elif macv2nat_aeronet_path and not os.path.exists(macv2nat_aeronet_path):
        raise ValueError(f"macv2nat_aeronet_path not found: {macv2nat_aeronet_path}")

    if merra2_aeronet_path is None:
        merra2_aeronet_path = get_merra2_aeronet_path()
    # Preliminary sanity checks
    elif merra2_aeronet_path != "" and not os.path.exists(merra2_aeronet_path):
        raise ValueError(f"Provided path {merra2_aeronet_path} not found.")

    if cams_free_aeronet_path is None:
        cams_free_aeronet_path = get_cams_free_aeronet_path(data_product="AOD")
    # Preliminary sanity checks
    elif isinstance(cams_free_aeronet_path, str) and not cams_free_aeronet_path == "":
        cams_free_aeronet_path = [cams_free_aeronet_path]
        allfiles = []
        for path in cams_free_aeronet_path:
            allfiles.extend(glob(path))
        if allfiles == []:
            raise ValueError(f"No files found for the given cams_free_aeronet_path(s).")

    # DEFAULTS vs overridden values
    if lon_to_degeast is None:
        lon_to_degeast = CONFIGDICT.get("lon_to_degeast", True)
    if t_approx_h is None:
        t_approx_h = CONFIGDICT.get("t_approx_h", True)

    if cams_clim_aeronet_path != "":
        if not os.path.exists(cams_clim_aeronet_path):
            print(f"CAMS climatology interpolated to Aeronet sites not found at {cams_clim_aeronet_path}. Will be generated")
            this_cams_clim_aeronet_path = None
        else:
            this_cams_clim_aeronet_path = cams_clim_aeronet_path
    else:
        this_cams_clim_aeronet_path = ""

    if macv2sp_aeronet_path != "":
        if not os.path.exists(macv2sp_aeronet_path):
            print(f"MACv2-SP data interpolated to Aeronet sites not found at {macv2sp_aeronet_path}. Will be generated")
            this_macv2sp_aeronet_path = None
        else:
            this_macv2sp_aeronet_path = macv2sp_aeronet_path
    else:
        this_macv2sp_aeronet_path = ""

    if macv2nat_aeronet_path != "":
        if not os.path.exists(macv2nat_aeronet_path):
            print(f"MACv2-SP natural not found at {macv2nat_aeronet_path}. Will generate.")
            this_macv2nat_aeronet_path = None
        else:
            this_macv2nat_aeronet_path = macv2nat_aeronet_path
    else:
        this_macv2nat_aeronet_path = ""

    if merra2_aeronet_path != "":
        if not os.path.exists(merra2_aeronet_path):
            print(f"MERRA2 data interpolated to Aeronet sites not found at {merra2_aeronet_path}. Will be generated")
            this_merra2_aeronet_path = None
        else:
            this_merra2_aeronet_path = merra2_aeronet_path
    else:
        this_merra2_aeronet_path = ""

    assert isinstance(lon_to_degeast, bool)
    assert isinstance(t_approx_h, bool)

    aeronet_coll = AeronetCollection(
        aeronet_dir_path=aeronet_sites_dir,
        cams_clim_path=this_cams_clim_aeronet_path,
        cams_free_path=cams_free_aeronet_path,
        macv2sp_path=this_macv2sp_aeronet_path,
        macv2nat_path=this_macv2nat_aeronet_path,
        merra2_path=this_merra2_aeronet_path,
        timerange=timerange
    )

    if this_cams_clim_aeronet_path is None:
        from src.utils.cams import gen_cams_pointinterp
        gen_cams_pointinterp(aeronet_coll, overwrite=overwrite)
        aeronet_coll.load_cams_clim(get_cams_clim_aeronet_path(data_product="AOD"))

    if this_macv2sp_aeronet_path is None:
        from src.utils.macv2sp import gen_macv2sp_pointinterp
        gen_macv2sp_pointinterp(macv2sp_points_fpath = get_macv2sp_aeronet_path(),
                                obs_coll = aeronet_coll,
                                overwrite = overwrite)
        aeronet_coll.load_macv2sp(get_macv2sp_aeronet_path())

    if this_macv2nat_aeronet_path is None:
        from src.utils.macv2sp import gen_macv2nat_pointinterp
        gen_macv2nat_pointinterp(macv2nat_points_fpath=get_macv2nat_aeronet_path(),
                                 obs_coll=aeronet_coll,
                                 overwrite=overwrite)
        aeronet_coll.load_macv2nat(get_macv2nat_aeronet_path())

    if this_merra2_aeronet_path is None:
        from src.utils.merra2 import gen_merra2_pointinterp
        gen_merra2_pointinterp(merra2_points_fpath = get_merra2_aeronet_path(),
                               obs_coll = aeronet_coll,
                               overwrite = overwrite)
        aeronet_coll.load_merra2(get_merra2_aeronet_path())

    return aeronet_coll

