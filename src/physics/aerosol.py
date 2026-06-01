from __future__ import annotations

import numpy as np
import xarray as xr

from .aerotools import IFSAeroSpecs, get_nccn_over_mcon_from_specs, get_mean_diam_from_spec

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

