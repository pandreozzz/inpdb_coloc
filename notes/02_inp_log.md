## Log
- Add collector function "get_cams_clim_inp" and "get_cams_free_inp" which makes the inp concentration available to a collection (given aerosol spec and inp parametrisation)
- Implement INPParametrisation class
    - Implement Harrison kf active site density parametrisation
    - Implement McCluskey marine organic active site density parametrisation
- Port Aerosol Spec for computing mean diameter and conversion factor from mass mixing ratio to number concentration per mass


## ToDo / Open tasks
- Derive pressure at given altitude. (e.g. from era5 fields)
    - Needed for density calculation, used for computing concentrations from mixing ratios
    - (And needed for co-location)
- Add option to use T from ERA5 for the INP parametrisation instead of T from inpdb. This would allow for plotting non-colocated model INP estimates