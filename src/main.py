from typing import Union, Optional, List, Tuple

from src.structs.collector import INPCollection

def build_collection(inpdb_csv_path : Optional[str] = None,
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
    """
    import os
    from glob import glob
    from src.config import get_inpdb_csv_path, get_cams_free_path, \
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
    elif not os.path.exists(cams_clim_sites_path):
        raise ValueError(f"Provided path {cams_clim_sites_path} not found.")
    
    if cams_free_sites_path is None:
        cams_free_sites_path = get_cams_free_path()
    # Preliminary sanity checks 
    elif isinstance(cams_free_sites_path, str):
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