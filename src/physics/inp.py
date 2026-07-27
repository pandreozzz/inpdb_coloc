from __future__ import annotations

from typing import Union, Optional, List, Tuple
import pandas as pd
import numpy as np
import xarray as xr

from collections.abc import Callable

from .constants import Constants

class INPParametrization:
        """Class to handle INP parametrizations. It should be able to take as input the co-located data and return INP concentrations per liter according to the parametrization."""
        def initialise(self,
                       site_density_function: Optional[Callable] = None,
                       aerosol_variables: Optional[List[str]] = None,
                       aerosol_fraction: Optional[float] = 1.0,
                       T_min: Optional[float] = None,
                       T_max: Optional[float] = None,
                       doi: Optional[str] = None
                       ):
            self.site_density_func = site_density_function
            self.aerovars = aerosol_variables
            self.aerosol_fraction = aerosol_fraction # Can be used to assume a fraction of the aerosol is active INP species, for example 5% feldspar of total dust
            self.T_min = T_min
            self.T_max = T_max
            self.doi = doi

        def compute_inp_concentration(self, temperature_data, air_density_data, cams_data, aerosol_spec, clip_to_T_range='clip_low_T'):
            """
            Compute INP concentration from the input data using the parametrization.
            By default the temperature data is clipped for temperatures below the range of validity of the parametrization.
            """

            inp_num_concs = {}
            for i, var in enumerate(self.aerovars):
                if var not in cams_data:
                    raise ValueError(f"Aerosol variable {var} not found in CAMS data.")
                aerosol_num_conc = cams_data[var] * aerosol_spec.get_num_conc_factor(var) * air_density_data
                mean_diameter = aerosol_spec.get_mean_diameter(var)
                if clip_to_T_range == "clip_both":
                    temperature_data = np.clip(temperature_data, self.T_min, self.T_max)
                elif clip_to_T_range == "clip_low_T":
                    # We don't want to artificially increase high temperature INP
                    temperature_data = np.clip(temperature_data, self.T_min, None)
                # Apply the aerosol fraction to the aerosol mass/number concentration (as in Herbert et al 2025)
                inp_num_conc = aerosol_num_conc * self.aerosol_fraction * (1 - np.exp(-self.site_density_func(temperature_data) * np.pi * mean_diameter ** 2)) * 1.e3
                inp_num_concs[var] = inp_num_conc

            return sum(inp_num_concs.values()) # Sum over the different aerosol bins to get total INP concentration


class INPParametrizationCatalog:
    """Class to store a catalog of INP parametrizations."""
    def __init__(self, parametrizations: Optional[dict[str, INPParametrization]] = None):
        self.parametrizations = {} if parametrizations is None else parametrizations
        self.load_parametrizations()

    def load_parametrizations(self):
        """Load the parametrizations into the catalog. This can be extended to load from files or other sources."""
        from .aerosol import AerosolSpec
        aerospec = AerosolSpec()

        # Harrison2019 parametrization for feldspar, which is the most active INP species in dust.
        # https://doi.org/10.5194/acp-19-11343-2019
        name = "Harrison2019"
        def ns_kf(t):
            tc = (t - 273.15)
            rs = -3.25 + (-0.793*tc) + (-6.91*1e-2*tc**2) +\
                + (-4.17*1e-3*tc**3) + (-1.05*1e-4*tc**4) +\
                + (-9.08*1e-7*tc**5)
            return 10**rs
        param = INPParametrization()
        param.initialise(
             site_density_function=ns_kf, # kfeldspar
             aerosol_variables=aerospec.dustvars,
             aerosol_fraction=0.05, # Assume 5% of dust is feldspar, which is the active INP species parametrised in Harrison et al. 2019
             T_min=-37.5 + 273.15, # Minimum temperature for the parametrization in Kelvin
             T_max=-3.5 + 273.15, # Maximum temperature for the parametrization in Kelvin
             doi="https://doi.org/10.5194/acp-19-11343-2019"
             )
        self.parametrizations[name] = param

        # Atkinson2013 parametrization for k-feldspar in dust. They report, feldspar mass content in natural soils typically varies between 1% and 25%
        # https://doi.org/10.1038/nature12278
        name = "Atkinson2013"
        def ns_kf(t):
            rs = -1.038*t + 275.26
            return np.exp(rs)
        param = INPParametrization()
        param.initialise(
             site_density_function=ns_kf, # kfeldspar
             aerosol_variables=aerospec.dustvars,
             aerosol_fraction=0.05, # Assume 5% of dust is feldspar (k-feldspar fraction); see Atkinson et al. 2013 for context
             T_max=268, # Maximum temperature for the parametrization in Kelvin
             doi="https://doi.org/10.1038/nature12278",
             )
        self.parametrizations[name] = param

        # O'Sullivan2014 parametrization for untreated soil dust, Parametrization in Fig. 8 caption
        # doi:10.5194/acp-14-1853-2014
        name = "OSullivan2014"
        def ns_soil(t):
            rs = 2.974*1.e-3*t**2 - 2.160*t + 366.3
            return np.exp(rs)
        param = INPParametrization()
        param.initialise(
             site_density_function=ns_soil, # untreated soil dust
             aerosol_variables=aerospec.dustvars,
             T_min=246, # Minimum temperature for the parametrization in Kelvin
             T_max=267, # Maximum temperature for the parametrization in Kelvin
             doi="https://doi.org/10.5194/acp-14-1853-2014",
             )
        self.parametrizations[name] = param

        # McCluskey2018 parametrization for marine organic based on sea salt aerosol, Parametrization in Fig. 8 caption
        # Nucleation site densities as a function of temperature measured in the CLEAN sector by the Ice Spectrometer
        # http://dx.doi.org/10.1029/2017JD028033
        name = "McCluskey2018"
        def ns_mom(t):
            a = -0.545
            b = 1.0125

            rs = a*(t - 273.15)+b

            return np.exp(rs)*1.e-4
        param = INPParametrization()
        param.initialise(
             site_density_function=ns_mom, # marine organic
             aerosol_variables=aerospec.marinvars,
             T_min=-28 + 273.15, # Minimum temperature for the parametrization in Kelvin – estimated from the data points shown in Figure 8 of McCluskey et al. 2018
             T_max=-10 + 273.15, # Maximum temperature for the parametrization in Kelvin – estimated from the data points shown in Figure 8 of McCluskey et al. 2018
             doi="http://dx.doi.org/10.1029/2017JD028033"
             )
        self.parametrizations[name] = param

    def get_parametrization(self, name: str) -> INPParametrization:
        """Get a parametrization by name."""
        if name not in self.parametrizations:
            raise ValueError(f"Parametrization {name} not found in catalog.")
        return self.parametrizations[name]
