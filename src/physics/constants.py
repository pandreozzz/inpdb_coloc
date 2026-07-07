from typing import Final

CONST_G : Final = 9.80655 # m/s^2 gravitational acceleration at sea level
CONST_Rd : Final = 287.0500676 # in J⋅kg−1⋅K−1 - specific gas constant for dry air
CONST_R : Final = 8.3144598 # J/(mol*K) universal gas constant


class Constants:
    """Class to handle physical constants."""
    def __init__(self):
        self.R_s = 287.0500676 # in J⋅kg−1⋅K−1 - specific gas constant for dry air
        self.M = 0.0289644 # kg/mol molar mass of air
        self.g = 9.80665 # m/s^2 gravitational acceleration at sea level
        self.R = 8.3144598 # J/(mol*K) universal gas constant

