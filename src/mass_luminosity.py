import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

'''
========================================================================
get_MS_mag(mass, dist, band, A=0)
========================================================================
Returns the magnitude of a main sequence star of a given mass at given distance in given band with (or without) extinction
We are using Eric Mamajek's table of main sequence stars, see: Pecault&Mamajek 2013 and Eric Mamajek's webpage
------------------------------------------------------------------------
Input:
mass -- mass of the main sequence star
dist -- distance to the star
band -- band for which we want to obtain the magnitude
A -- default 0, extinction in given band towards our star
'''


def initialize_interp(band: str):
    mag = []
    mass = []
    # First download the table
    url = r'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
    mass_lum = pd.read_csv(url, delim_whitespace=True, nrows=118, skiprows=21, header=1)
    # For given filter get the masses and corresponding abolute magnitudes
    if band == "V":
        # absolute magnitude in V
        mag_v = mass_lum.loc[
            (mass_lum["Mv"] != "...") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "Mv"].values
        # corresponding mass
        mass = mass_lum.loc[
            (mass_lum["Mv"] != "...") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "Msun"].values
        mag, mass = mag_v.astype(float), mass.astype(float)
    elif band == "I":
        # absolute magnitude in V
        mag_v = mass_lum.loc[
            (mass_lum["Mv"] != "...") & (mass_lum["V-Ic"] != "...") & (mass_lum["V-Ic"] != ".....") & (
                    mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")]["Mv"].values
        # corresponding colour in V-Ic
        v_i = mass_lum.loc[
            (mass_lum["Mv"] != "...") & (mass_lum["V-Ic"] != "...") & (mass_lum["V-Ic"] != ".....") & (
                    mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")]["V-Ic"].values
        # corresponding mass
        mass = mass_lum.loc[
            (mass_lum["Mv"] != "...") & (mass_lum["V-Ic"] != "...") & (mass_lum["V-Ic"] != ".....") & (
                    mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")]["Msun"].values
        mag_v, v_i, mass = mag_v.astype(float), v_i.astype(float), mass.astype(float)
        # absolute magnitude in I, calculated using provided abs mag in V and colour in V-Ic
        mag = mag_v - v_i
    elif band == "G":
        # absolute magnitude in V
        mag_g = mass_lum.loc[
            (mass_lum["M_G"] != "...") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "M_G"].values
        # corresponding mass
        mass = mass_lum.loc[
            (mass_lum["M_G"] != "...") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "Msun"].values
        mag, mass = mag_g.astype(float), mass.astype(float)
    elif band == "K":
        # absolute magnitude in K
        mag_g = mass_lum.loc[
            (mass_lum["M_Ks"] != "...") & (mass_lum["M_Ks"] != "....") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "M_Ks"].values
        # corresponding mass
        mass = mass_lum.loc[
            (mass_lum["M_Ks"] != "...") &  (mass_lum["M_Ks"] != "....") & (mass_lum["Msun"] != "....") & (mass_lum["Msun"] != "...")][
            "Msun"].values
        mag, mass = mag_g.astype(float), mass.astype(float)
    else:
        print("Mamajek problem: requested filter does not exist")
        exit(2)

    mag_min = float(mag[0])  # minimal abs. magnitude for which mass is known
    mag_max = float(mag[-1])  # maximal abs. magnitude for which mass is known
    fun = interp1d(mass, mag, kind='linear')  # interpolation function for masses between known
    return fun, mag_min, mag_max


def get_ms_mag(fun, mass: float, dist: float, band: str, extinction: float, mag_min: float, mag_max: float) -> float:
    # Find the observed magnitude of the main sequence star
    if band == 'V':
        if mass > 27.:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
        elif mass < 0.075:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
        else:
            mag = fun(mass) + 5 * np.log10(dist * 1000.) - 5 + extinction
    elif band == 'I' or band == 'K':
        if mass > 19.8:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
        elif mass < 0.075:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
        else:
            mag = fun(mass) + 5 * np.log10(dist * 1000.) - 5 + extinction
    elif band == 'G':
        if mass > 5.4:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
        elif mass < 0.075:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
        else:
            mag = fun(mass) + 5 * np.log10(dist * 1000.) - 5 + extinction
    else:
        return 0

    return mag

def get_abs_mag_g_ms_malkov(mass, mag_min, mag_max):
    '''
    This function returns an absolute magnitude for a MS star
    using the relation from Malkov et al. 2022 (https://ui.adsabs.harvard.edu/abs/2022Ap%26SS.367...37M/abstract).

    Args:
        mass: Mass of the lens.
        mag_min: Minimum absolute magnitude when mass exceeds Mamajek's table range
        mag_max: Maximum absolute magnitude when mass exceeds Mamajek's table range

    Returns:
        Absolute magnitude in G band of an MS star.

    '''
    abs_mag_g = np.nan

    if((18. > mass) and (mass > 2.27)):
        abs_mag_g = -5.67 * np.log10(mass) + 2.96
    elif((2.27 > mass) and (mass > 1.4)):
        abs_mag_g = -9.53 * np.log10(mass) + 4.33
    elif(mass > 18.):
        abs_mag_g = mag_min
    elif(mass < 1.4):
        abs_mag_g = mag_max

    return abs_mag_g


def get_abs_mag_g_subgiant_malkov(mass, mag_min, mag_max):
    '''
    This function returns an absolute magnitude for a subgiant star
    using the relation from Malkov et al. 2022 (https://ui.adsabs.harvard.edu/abs/2022Ap%26SS.367...37M/abstract).

    Args:
        mass: Mass of the lens.
        mag_min: Minimum absolute magnitude when mass exceeds Malkov et al. relation range
        mag_max: Maximum absolute magnitude when mass exceeds Malkov et al. relation range

    Returns:
        Absolute magnitude in G band of a subgiant star.

    '''
    abs_mag_g = np.nan

    if ((18. > mass) and (mass > 3.84)):
        abs_mag_g = -6.74 * np.log10(mass) + 2.46
    elif ((3.84 > mass) and (mass > 1.4)):
        abs_mag_g = -8.98 * np.log10(mass) + 3.77
    elif (mass > 18.):
        abs_mag_g = mag_min
    elif (mass < 1.4):
        abs_mag_g = mag_max

    return abs_mag_g

def get_mag_g_malkov(mass, distance, extinction, subgiant_flag=False):
    if(subgiant_flag):
        mag_min = -6.00
        mag_max = 2.46
        abs_mag = get_abs_mag_g_subgiant_malkov(mass, mag_min, mag_max)
    else:
        mag_min = -4.16
        mag_max = 2.94
        abs_mag = get_abs_mag_g_ms_malkov(mass, mag_min, mag_max)

    mag = abs_mag + 5. * np.log10(dist * 1000.) - 5. + extinction

    return mag
