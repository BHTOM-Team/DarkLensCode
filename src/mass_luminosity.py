import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import Stefan_Boltzmann as sigma_SB
from astropy.constants import R_sun, L_sun

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
    mass_lum = pd.read_csv(url, sep='\s+', nrows=118, skiprows=21, header=1)
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

def initialize_interp_radius_dwarf(band: str):
    '''
    Function initializng radius and absolute magnitude relation.
    Cite Pecalut&Mamajek 2013, linkt from Eric Mamajek's personal page.
    '''
    mag = []
    radius = []
    # First download the table
    url = r'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
    data = pd.read_csv(url, sep='\s+', nrows=92, skiprows=21, header=1)
    # print(data)
    # For given filter get the masses and corresponding abolute magnitudes
    if band == "V":
        # absolute magnitude in V
        mag_v = data.loc[
            (data["Mv"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "Mv"].values
        # corresponding radius
        radius_v = data.loc[
            (data["Mv"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values
        mag, radius = mag_v.astype(float), radius_v.astype(float)
    elif band == "I":
        # absolute magnitude in V
        mag_v = data.loc[
            (data["Mv"] != "...") & (data["V-Ic"] != "...") & (data["V-Ic"] != ".....") & (
                    data["Msun"] != "....") & (data["Msun"] != "...")][
            "Mv"].values
        # corresponding colour in V-Ic
        v_i = data.loc[
            (data["Mv"] != "...") & (data["V-Ic"] != "...") & (data["V-Ic"] != ".....") & (
                    data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")]["V-Ic"].values
        # corresponding radius
        radius_v = data.loc[
            (data["Mv"] != "...") & (data["V-Ic"] != "...") & (data["V-Ic"] != ".....") & (
                    data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values

        mag_v, v_i, radius = mag_v.astype(float), v_i.astype(float), radius_v.astype(float)
        # absolute magnitude in I, calculated using provided abs mag in V and colour in V-Ic
        mag = mag_v - v_i
    elif band == "G":
        # absolute magnitude in V
        mag_g = data.loc[
            (data["M_G"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "M_G"].values
        # corresponding mass
        radius_g = data.loc[
            (data["M_G"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values
        mag, radius = mag_g.astype(float), radius_g.astype(float)
    elif band == "K":
        # absolute magnitude in K
        mag_k = data.loc[
            (data["M_Ks"] != "...") & (data["M_Ks"] != "....") & (data["R_Rsun"] != "....") & (
                        data["R_Rsun"] != "...")][
            "M_Ks"].values
        # corresponding mass
        radius_k = data.loc[
            (data["M_Ks"] != "...") & (data["M_Ks"] != "....") & (data["R_Rsun"] != "....") & (
                        data["R_Rsun"] != "...")][
            "R_Rsun"].values
        mag, radius = mag_k.astype(float), radius_k.astype(float)
    else:
        print("Mamajek problem: requested filter does not exist")
        exit(2)

    mag_min = float(mag[0])  # minimal abs. magnitude for which mass is known
    mag_max = float(mag[-1])  # maximal abs. magnitude for which mass is known
    fun = interp1d(radius, mag, kind='linear')  # interpolation function for radii between known
    return fun, mag_min, mag_max

def initialize_interp_teff_dwarf(band: str):
    '''
    Function initializng radius and effective temperature relation.
    Cite Pecalut&Mamajek 2013, linkt from Eric Mamajek's personal page.
    '''
    mag = []
    radius = []
    # First download the table
    url = r'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
    data = pd.read_csv(url, sep='\s+', nrows=92, skiprows=21, header=1)
    # print(data)
    # For given filter get the masses and corresponding abolute magnitudes
    if band == "V":
        # absolute magnitude in V
        t_eff_v = data.loc[
            (data["Mv"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "Teff"].values
        # corresponding radius
        radius_v = data.loc[
            (data["Mv"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values
        t_eff, radius = t_eff_v.astype(float), radius_v.astype(float)
    elif band == "I":
        # absolute magnitude in V
        t_eff_i = data.loc[
            (data["Mv"] != "...") & (data["V-Ic"] != "...") & (data["V-Ic"] != ".....") & (
                    data["Msun"] != "....") & (data["Msun"] != "...")][
            "Teff"].values
        # corresponding radius
        radius_i = data.loc[
            (data["Mv"] != "...") & (data["V-Ic"] != "...") & (data["V-Ic"] != ".....") & (
                    data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values

        t_eff, radius = t_eff_i.astype(float), radius_i.astype(float)
    elif band == "G":
        # absolute magnitude in V
        t_eff_g = data.loc[
            (data["M_G"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "Teff"].values
        # corresponding mass
        radius_g = data.loc[
            (data["M_G"] != "...") & (data["R_Rsun"] != "....") & (data["R_Rsun"] != "...")][
            "R_Rsun"].values
        t_eff, radius = t_eff_g.astype(float), radius_g.astype(float)
    elif band == "K":
        # absolute magnitude in K
        t_eff_k = data.loc[
            (data["M_Ks"] != "...") & (data["M_Ks"] != "....") & (data["R_Rsun"] != "....") & (
                        data["R_Rsun"] != "...")][
            "Teff"].values
        # corresponding mass
        radius_k = data.loc[
            (data["M_Ks"] != "...") & (data["M_Ks"] != "....") & (data["R_Rsun"] != "....") & (
                        data["R_Rsun"] != "...")][
            "R_Rsun"].values
        t_eff, radius = t_eff_k.astype(float), radius_k.astype(float)
    else:
        print("Mamajek problem: requested filter does not exist")
        exit(2)

    t_min = float(t_eff[-1])  # minimal abs. magnitude for which mass is known
    t_max = float(t_eff[0])  # maximal abs. magnitude for which mass is known
    fun = interp1d(radius, t_eff, kind='linear')  # interpolation function for radii between known
    return fun, t_min, t_max

def get_dwarf_obs_mag(fun, fun_teff, radius: float, err_radius: float, dist: float, band: str, extinction: float, mag_min: float, mag_max: float, teff_min: float, teff_max: float,) -> float:
    # Find the observed magnitude of the main sequence star
    if band == 'V':
        if radius > 13.43:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = 13.43 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_max
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        elif radius < 0.0940:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = 0.0940 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_min
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        else:
            mag = fun(radius) + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = radius * R_sun.value  # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = fun_teff(radius)
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
    elif band == 'I' or band == 'K':
        if radius > 7.72:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = 7.72 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_max
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        elif radius < 0.0940:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = 0.0940 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_min
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        else:
            mag = fun(radius) + 5 * np.log10(dist * 1000.) - 5 + extinction
            rad_in_m = radius * R_sun.value  # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = fun_teff(radius)
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
    elif band == 'G':
        if radius > 3.61:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_min + 5 * np.log10(dist * 1000.) - 5 + extinction
            # print(mag_min)
            rad_in_m = 3.61 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_max
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        elif radius < 0.0940:
            # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
            mag = mag_max + 5 * np.log10(dist * 1000.) - 5 + extinction
            # print(mag_max)
            rad_in_m = 0.0940 * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = teff_min
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4)
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum
        else:
            mag = fun(radius) + 5 * np.log10(dist * 1000.) - 5 + extinction
            # print(fun(radius))
            rad_in_m = radius * R_sun.value # in meters to work with Stefan-Boltzmann constant
            err_rad_m = err_radius * R_sun.value
            t_eff = fun_teff(radius)
            lum = 4 * np.pi * sigma_SB * rad_in_m ** 2 * t_eff ** 4 / L_sun.value
            err_lum = (8 * np.pi * rad_in_m * err_rad_m * sigma_SB * t_eff ** 4 )
            err_lum = err_lum / L_sun.value
            err_mag = 2.5 * np.log(10) * err_lum / lum

    else:
        return 0

    # print("radius:", radius, "dist:", dist, "mag_obs: ", mag)
    return mag, err_mag

def get_giant_obs_mag_g(radius: float, err_radius: float, dist: float, extinction: float) -> float:
    '''
    Produces observed magnitude for a giant star in G band!!!!
    Requires radius of the star in solar radii,
    distance to the star,
    extinction in G band.
    Radius vs (V_0-K_0) relation van Belle et al. 2021, https://ui.adsabs.harvard.edu/abs/2021ApJ...922..163V/abstract
    (V_0-K_0) vs T_eff relation van Belle et al. 2021, https://ui.adsabs.harvard.edu/abs/2021ApJ...922..163V/abstract
    Bolometric correction for Gaia from GDR2:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu8par/sec_cu8par_process/ssec_cu8par_process_flame.html
    '''
    # Find corresponding V_0-K_0 using van Belle et al. 2021, table 16
    # I inverted the relation
    R_s = np.array([9.25, 13.40, 23.60, 31.25, 58.5, 67.55, 92.9, 110.45, 120.50, 180.8])
    val_a = [1.3, 26.5, 20.4, 53.5, 44.4, 17.9, 29.2, 7., 20.1]
    val_b = [7.3, -56.2, -37.6, -156., -125., -13.0, -69.0, 57.9, -0.1]

    if (radius > 180.8):
        # print("WARN radius exceeds van Belle range ", mass, dist, extinction)
        V_K_0 = (R_s[-1] - val_b[-1]) / val_a[-1]
        err_VK = err_radius / val_a[-1]
    elif (radius<9.25):
        # print("WARN radius exceeds van Belle range ", mass, dist, extinction)
        V_K_0 = (R_s[0] - val_b[0])/ val_a[0]
        err_VK = err_radius / val_a[0]
    else:
        idx = np.where(radius >= R_s)[0][-1]
        V_K_0 = (radius - val_b[idx]) / val_a[idx]
        err_VK = err_radius / val_a[idx]

    # Find effective temperature of the star
    # from van Belle et al. 2021
    T_eff = (-9.301 * V_K_0**3) + (212.4 * V_K_0**2) + (-1648 * V_K_0) + 7602
    err_Teff = err_VK * np.abs((3 * (-9.301) * V_K_0**2) + (2* 212.4 * V_K_0) - 1648)

    # Find bolometric correction -> https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu8par/sec_cu8par_process/ssec_cu8par_process_flame.html
    coeffs_8000 = np.array([6.e-02, 6.731e-05, -6.647e-08, 2.859e-11, -7.197e-15])
    coeffs_4000 = np.array([1.749, 1.977e-03, 3.737e-07, -8.966e-11, -4.183e-14])

    BC = 0.
    err_BC = 0.
    if (T_eff < 3300):
        # print("WARN T_eff exceeds GDR2 correction range ", mass, dist, extinction)
        T_eff = 3300.
    elif (T_eff > 8000):
        # print("WARN T_eff exceeds GDR2 correction range ", mass, dist, extinction)
        T_eff = 8000.

    if ((T_eff >= 3300) and (T_eff <= 4000)):
        for i in range(4):
            BC += coeffs_4000[i] * (T_eff - 5772.)**i # T_eff,sun = 5772
        err_BC = err_Teff * (coeffs_4000[1] +
                             (2 * coeffs_4000[2]) +
                             (3 * coeffs_4000[3] * (T_eff - 5772.) ** 2) +
                             (4 * coeffs_4000[4] * (T_eff - 5772.) ** 3)
                             )
    else:
        for i in range(4):
            BC += coeffs_8000[i] * (T_eff - 5772.) ** i
        err_BC = err_Teff * (coeffs_8000[1] +
                             (2 * coeffs_8000[2]) +
                             (3 * coeffs_8000[3] * (T_eff - 5772.) ** 2) +
                             (4 * coeffs_8000[4] * (T_eff - 5772.) ** 3)
                             )

    # Find bolometric lumninosity
    bol_lum = 0.
    # Bolometric luminosity has to be in L_sun, otherwise we get wrong units
    if (radius > 180.8):
        # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
        rad_in_m = R_s[-1] * R_sun.value # in meters to work with Stefan-Boltzmann constant
        bol_lum = 4 * np.pi * rad_in_m**2 * sigma_SB * T_eff**4 / L_sun.value
        err_bol_lum = ((4 * np.pi * 2 * rad_in_m * sigma_SB* T_eff**4 * err_radius) +
                       (4 * np.pi * rad_in_m**2 * sigma_SB * 4 * T_eff**3 * err_Teff))
        err_bol_lum = err_bol_lum / L_sun.value
    elif (radius<9.25):
        # print("WARN Mass exceeds Mamajek range ", mass, dist, extinction)
        rad_in_m = R_s[0] * R_sun.value # in meters to work with Stefan-Boltzmann constant
        bol_lum = 4 * np.pi * rad_in_m ** 2 * sigma_SB * T_eff ** 4 / L_sun.value
        err_bol_lum = ((4 * np.pi * 2 * rad_in_m * sigma_SB * T_eff ** 4 * err_radius) +
                   (4 * np.pi * rad_in_m ** 2 * sigma_SB * 4 * T_eff ** 3 * err_Teff))
        err_bol_lum = err_bol_lum / L_sun.value
    else:
        rad_in_m = radius * R_sun.value # in meters to work with Stefan-Boltzmann constant
        bol_lum = 4 * np.pi * rad_in_m**2 * sigma_SB * T_eff**4 / L_sun.value
        err_bol_lum = ((4 * np.pi * 2 * rad_in_m * sigma_SB * T_eff ** 4 * err_radius) +
                   (4 * np.pi * rad_in_m ** 2 * sigma_SB * 4 * T_eff ** 3 * err_Teff))
        err_bol_lum = err_bol_lum / L_sun.value

    abs_mag = 4.74 - (2.5 * np.log10(bol_lum)) - BC
    err_abs_mag = (2.5 * np.log(10) * err_bol_lum/ bol_lum) + err_BC

    mag = abs_mag + 5 * np.log10(dist * 1000.) - 5 + extinction
    err_mag = err_abs_mag
    return mag, err_mag

def initialize_extinction(extinction_params):
    '''

    Args:
        mass_power_params: String holding extinction coefficients from yaml file

    Returns:
        an array with coefficients

    '''
    params = extinction_params.split(',')
    length = int(len(params))
    coeffes = np.zeros(length)

    for i in range(length):
        coeffes[i] = float(params[i])

    return coeffes