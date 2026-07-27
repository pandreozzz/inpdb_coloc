from __future__ import annotations

import numpy as np
import xarray as xr

from .aerotools import IFSAeroSpecs, get_nccn_over_mcon_from_specs, get_mean_diam_from_spec


class AerosolSpec:
    """Class to handle aerosol specifications."""
    def __init__(self):
        self.ifsaero = self.set_default_IFSAERO()
        self.dustvars  = [f"Mineral_Dust_bin{i}" for i in (1,2,3)]
        self.marinvars = [f"Sea_Salt_bin{i}" for i in (1,2,3)]
        self.carbvars = [f"Black_Carbon_hydroph{xx}" for xx in ["ilic", "obic"]]
        self.aerovars = self.dustvars + self.marinvars + self.carbvars
        pass

    def get_num_conc_factor(self, var):
        """Get the factor to convert mass mixing ratio to number concentration for a given aerosol species."""
        aerospec = self.ifsaero[var]

        nccn_over_mcons = get_nccn_over_mcon_from_specs(aerospec)
        return nccn_over_mcons

    def get_num_conc_factor_all(self):
        """Get the factor to convert mass mixing ratio to number concentration for all aerosol species."""
        aerospecs = {v: self.ifsaero[v] for v in self.aerovars}

        nccn_over_mcons = xr.Dataset(
                data_vars = {v: get_nccn_over_mcon_from_specs(spec)
                            for v,spec in aerospecs.items()}
            )
        return nccn_over_mcons

    def get_mean_diameter(self, var):
        """Get the mean diameter of the aerosol particles for a given aerosol species."""
        aerospec = self.ifsaero[var]
        return get_mean_diam_from_spec(aerospec)*1.e-4

    def get_mean_diameter_all(self):
        """Get the mean diameter of the aerosol particles for all aerosol species."""
        aerospecs = {v: self.ifsaero[v] for v in self.aerovars}
        return {v: get_mean_diam_from_spec(spec)*1.e-4 for v, spec in aerospecs.items()}

    def set_default_IFSAERO(self):
        """Set the default IFS aerosol specifications."""
        # Numbers per each dust distribution mode
        ns_dust = np.array([391,8.39,11.6,0.000138])
        dust_factors = tuple(ns_dust/ns_dust.sum())

        IFSAERO = {sp.name : sp  for sp in [
            IFSAeroSpecs("Sea_Salt_bin1",
                        (0.1002,  1.002),
                        (1.9, 2.0),
                        2180,
                        0.0151,0.251,
                        (0.96, 0.04),
                        ),
            IFSAeroSpecs("Sea_Salt_bin2",
                        (0.1002,  1.002),
                        (1.9, 2.0),
                        2180,
                        0.251,2.51,
                        (0.96, 0.04),
                        ),
            IFSAeroSpecs("Sea_Salt_bin3",
                        (0.1002,  1.002),
                        (1.9, 2.0),
                        2180,
                        2.51,10.6,
                        (0.96, 0.04),
                        ),
            IFSAeroSpecs("Mineral_Dust_bin1",
                        (0.05,0.42,0.79,16.2),
                        (2.2,1.18,1.93,1.53),
                        2610,
                        0.03,0.55,
                        dust_factors,
                        ),
            IFSAeroSpecs("Mineral_Dust_bin2",
                        (0.05,0.42,0.79,16.2),
                        (2.2,1.18,1.93,1.53),
                        2610,
                        0.55,0.9,
                        dust_factors,
                        ),
            IFSAeroSpecs("Mineral_Dust_bin3",
                        (0.05,0.42,0.79,16.2),
                        (2.2,1.18,1.93,1.53),
                        2610,
                        0.9,20,
                        dust_factors,
                        ),
            IFSAeroSpecs("Black_Carbon_hydrophobic",
                        (0.0118,),
                        (2.0,),
                        1000,
                        0.005, 0.5),
            IFSAeroSpecs("Black_Carbon_hydrophilic",
                        (0.0118,),
                        (2.0,),
                        1000,
                        0.005, 0.5),
            IFSAeroSpecs("Organic_Matter_hydrophobic",
                        (0.09,),
                        (1.6,),
                        1300,
                        0.005, 20),
            IFSAeroSpecs("Organic_Matter_hydrophilic",
                        (0.09,),
                        (1.6,),
                        1300,
                        0.005, 20),
            IFSAeroSpecs("Biogenic_Secondary_Organic",
                        (0.09,),
                        (1.6,),
                        1800,
                        0.005, 20),
            IFSAeroSpecs("Anthropogenic_Secondary_Organic",
                        (0.09,),
                        (1.6,),
                        1800,
                        0.005, 20),
            IFSAeroSpecs("Sulfates",
                        (0.11,),
                        (1.6,),
                        1760,
                        0.005, 20),
            IFSAeroSpecs("Ammonium",
                        (0.0355,),
                        (2.0,),
                        1760,
                        0.005, 20),
            IFSAeroSpecs("Nitrate_fine",
                        (0.0355,),
                        (2.0,),
                        1730,
                        #0.005, 20),
                        0.03, 0.9),
            IFSAeroSpecs("Nitrate_coarse",
                        (0.199, 1.992),
                        (1.9, 2.0),
                        1400,
                        0.9, 20,
                        (0.96, 0.04)
                    ),
        ]}

        return IFSAERO


# -----------------------------------------------------------------------------
# IFS aerosol optics index map (copied from cdnc_toolbox/v4/aeroptics.py)
# -----------------------------------------------------------------------------
# Maps an aerosol species name to ``(optics_index, is_hydrophilic)`` for the
# IFS aerosol optics LUT. The optics_index is 1-based and selects the row of the
# ``hydrophilic`` / ``hydrophobic`` axis of the LUT (see PHILIC_DESCR /
# PHOBIC_DESCR below); ``is_hydrophilic`` selects which axis to use.
PHILIC_DESCR = \
"""
1: Sea salt, bin 1, 0.03-0.5 micron, OPAC
2: Sea salt, bin 2, 0.50-5.0 micron, OPAC
3: Sea salt, bin 3, 5.0-20.0 micron, OPAC
4: Hydrophilic organic matter, OPAC
5: Ammonium sulfate (for sulfate), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
6: Secondary organic aerosol - biogenic, Moise et al 2015
7: Secondary organic aerosol - anthropogenic, Moise et al 2015
8: Fine mode Ammonium sulfate (for ammonia), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
9: Fine mode Nitrate, GLOMAP
10: Coarse mode Nitrate, GLOMAP
11: Hydrophilic organic matter, Brown et al 2018
12: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
13: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/ with modified size distribution
14: Desert dust, bin 1, 0.03-0.55 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
15: Desert dust, bin 2, 0.55-0.90 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
16: Desert dust, bin 3, 0.90-20.0 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
17: Desert dust, bin 1, 0.03-0.55 micron, Composite-Philic (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
18: Desert dust, bin 2, 0.55-0.90 micron, Composite-Philic (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
19: Desert dust, bin 3, 0.90-20.0 micron, Composite-Philic (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
"""
PHOBIC_DESCR = \
"""
1: Desert dust, bin 1, 0.03-0.55 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
2: Desert dust, bin 2, 0.55-0.90 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
3: Desert dust, bin 3, 0.90-20.0 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
4: Desert dust, bin 1, 0.03-0.55 micron, Fouquart et al 1987
5: Desert dust, bin 2, 0.55-0.90 micron, Fouquart et al 1987
6: Desert dust, bin 3, 0.90-20.0 micron, Fouquart et al 1987
7: Desert dust, bin 1, 0.03-0.55 micron, Woodward 2001, Table 2
8: Desert dust, bin 2, 0.55-0.90 micron, Woodward 2001, Table 2
9: Desert dust, bin 3, 0.90-20.0 micron, Woodward 2001, Table 2
10: Hydrophobic organic matter, OPAC (hydrophilic at RH=20%)
11: Black carbon, OPAC
12: Black carbon, Bond and Bergstrom 2006
13: Black carbon, Stier et al 2007
14: Stratospheric sulfate (hydrophilic ammonium sulfate at RH 20%-30%)
15: Desert dust, bin 1, 0.03-0.55 micron, Composite (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
16: Desert dust, bin 2, 0.55-0.90 micron, Composite (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
17: Desert dust, bin 3, 0.90-20.0 micron, Composite (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)
19: Black carbon, Williams 2007
"""

OPTICS_AERO_MAP: dict = {}

###
## Prognostic 43r3 and Bozzo climatology
###
OPTICS_AERO_MAP["43r3"] = {
    "Sea_Salt_bin1"                   : (1, True),
    "Sea_Salt_bin2"                   : (2, True),
    "Sea_Salt_bin3"                   : (3, True),
    "Mineral_Dust_bin1"               : (7, False), # Composite-Phobic
    "Mineral_Dust_bin2"               : (8, False), # Composite-Phobic
    "Mineral_Dust_bin3"               : (9, False), # Composite-Phobic
    "Organic_Matter_hydrophilic"      : (4, True),
    "Organic_Matter_hydrophobic"      : (4, False),
    "Black_Carbon_hydrophilic"        : (11, False),
    "Black_Carbon_hydrophobic"        : (11, False),
    "Sulfates"                        : (5, True)
}

###
## Prognostic 48r1
###
OPTICS_AERO_MAP["48r1"] = {
    **OPTICS_AERO_MAP["43r3"].copy(),
    **{
    "Nitrate_fine"                    : (9, True),
    "Nitrate_coarse"                  : (10, True),
    "Ammonium"                        : (8, True),
    "Biogenic_Secondary_Organic"      : (6, True),
    "Anthropogenic_Secondary_Organic" : (7, True),
    "Stratospheric_Sulfate"           : (14, False),
    }
}
# Composite phobic dust
OPTICS_AERO_MAP["48r1"]["Mineral_Dust_bin1"] = (15, False)
OPTICS_AERO_MAP["48r1"]["Mineral_Dust_bin2"] = (16, False)
OPTICS_AERO_MAP["48r1"]["Mineral_Dust_bin3"] = (17, False)
# Brown OM
OPTICS_AERO_MAP["48r1"]["Organic_Matter_hydrophilic"] = (11, True)
OPTICS_AERO_MAP["48r1"]["Organic_Matter_hydrophobic"] = (10, False)

###
## IFS-COMPO 48r1-based 4D climatology (Tim's) deployed in IFS 49R2
## has inconsistent sulfates and uses the new PSD
###
OPTICS_AERO_MAP["48r1_4dclim"]  = OPTICS_AERO_MAP["48r1"].copy()
OPTICS_AERO_MAP["48r1_4dclim"]["Sulfates"] = (13, True)

###
## Prognostic 49r1
###
OPTICS_AERO_MAP["49r1"] = OPTICS_AERO_MAP["48r1"].copy()

# New PSD for Sulfates
OPTICS_AERO_MAP["48r1_4dclim"]["Sulfates"] = (13, True)

# Hydrophilic Dust
OPTICS_AERO_MAP["49r1"]["Mineral_Dust_bin1"] = (14, True)
OPTICS_AERO_MAP["49r1"]["Mineral_Dust_bin2"] = (15, True)
OPTICS_AERO_MAP["49r1"]["Mineral_Dust_bin3"] = (16, True)

# Bond BC
OPTICS_AERO_MAP["49r1"]["Black_Carbon_hydrophobic"] = (12, False)
OPTICS_AERO_MAP["49r1"]["Black_Carbon_hydrophilic"] = (12, False)

