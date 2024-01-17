# a simple script for converting lensing parameters obtained
# in the geocentric frame of reference to the heliocentric frame of reference.
#
# ZK
# z.kaczmarek@student.uw.edu.pl

import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import math
import matplotlib.pyplot as plt

def get_dirs(ra_deg, dec_deg):

    """
    Returns local 3D north, east directions for a given point on the sky.

    Args:
        - ra_deg, dec_deg    float - coordinates on the sky, deg
    Returns:
        - north    ndarray - north (increasing dec) direction in xyz coords
        - east     ndarray - east (increasing ra) direction in xyz coords
    """

    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)

    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)])

    return east, north


def get_Earth_pos(t0par, alpha, delta):

    """
    Returns Earth velocity and position projected on the sky for a given event position and reference time.

    Args:
        - t0par     float - reference time used for modelling, JD-2450000
        - alpha, delta    float - coordinates of the event, deg
    Returns:
        - x_Earth_perp    ndarray - projected NE Earth position at the defined time and direction, au
        - v_Earth_perp    ndarray - projected NE Earth velocity at the defined time and direction, au/d
    """

    east, north = get_dirs(alpha, delta)

    t0par_jd = Time(t0par + 2450000, format='jd')
    x_Earth_3D, v_Earth_3D = get_body_barycentric_posvel('earth', t0par_jd)
    #unpack from astropy CartesianRepresentation
    x_Earth_3D = np.array([x_Earth_3D.x.value, x_Earth_3D.y.value, x_Earth_3D.z.value])
    v_Earth_3D = np.array([v_Earth_3D.x.value, v_Earth_3D.y.value, v_Earth_3D.z.value])


    x_earth_N = np.dot(x_Earth_3D, north)
    x_earth_E = np.dot(x_Earth_3D, east)
    v_earth_N = np.dot(v_Earth_3D, north)
    v_earth_E = np.dot(v_Earth_3D, east)

    x_Earth_perp = np.array([x_earth_N, x_earth_E])
    v_Earth_perp = np.array([v_earth_N, v_earth_E])

    return x_Earth_perp, v_Earth_perp

def get_angle(vec1, vec2):

    """
    A very simple function taking two 2D vectors and returning an angle between them.
    """

    ang = math.atan2(vec2[1], vec2[0]) - math.atan2(vec1[1], vec1[0])
    return ang

def get_helio_params(t0_geo, tE_geo, u0_geo, piEN_geo, piEE_geo, t0par, alpha, delta):

    """
    Obtains heliocentric parameters using analogous geocentric parameters, reference time and coordinates.

    Args:
        -----
        standard output from modelling, all defined in the geocentric reference frame:
        - t0_geo - time of maximum approach on the straight line trajectory, JD-2450000
        - tE_geo - timescale of the event, days
        - u0_geo - impact parameter, units of thetaE
        - piEN_geo, piEE_geo - components of the parallax vector, units of thetaE
        -----
        - t0par     float - reference time used for modelling, JD-2450000
        - alpha, delta    float - coordinates of the event, deg

    Returns:
        - t0_helio, tE_helio, u0_helio, piEN_helio, piEE_geo - analogous to input parameters with a 'geo' index,
        defined in the heliocentric reference frame

    """

    piE_geo = [piEN_geo, piEE_geo]
    piE = np.linalg.norm(piE_geo)

    x_Earth_perp, v_Earth_perp = get_Earth_pos(t0par, alpha, delta)
    x_Earth_perp = x_Earth_perp*piE

    x_Earth_perp_norm = np.linalg.norm(x_Earth_perp)
    v_Earth_perp_norm = np.linalg.norm(v_Earth_perp)

    tE_helio_inverse = (1/tE_geo)*(piE_geo/piE) + v_Earth_perp*piE
    piE_helio = piE * tE_helio_inverse/np.linalg.norm(tE_helio_inverse)
    piEN_helio, piEE_helio = piE_helio[0], piE_helio[1]

    tE_helio = 1/np.linalg.norm(tE_helio_inverse)

    # direction of x_Earth_perp in (tau, beta) coords
    phi = get_angle(piE_geo, -x_Earth_perp)
    # direction of helio motion in (tau, beta) coords
    delta_phi = get_angle(piE_geo, piE_helio)

    # x_Earth_perp expressed in (tau, beta) coords
    x_Earth_perp_taub = np.array([x_Earth_perp_norm*np.cos(phi) , x_Earth_perp_norm*np.sin(phi)])

    # projected Sun, Earth positions at t0par
    vecEartht0par = [-(t0_geo - t0par)/tE_geo, u0_geo]
    vecSunt0par = vecEartht0par - x_Earth_perp_taub

    # projecting Sun position on the direction of helio source-lens motion
    l = -vecSunt0par
    v = np.array([np.cos(delta_phi), np.sin(delta_phi)])
    c = np.dot(l,v)/np.dot(v,v)

    t0_helio = t0par + c*tE_helio
    u_vec = l - c*v

    # final correction for u0 sign
    sgn_u = 1
    if((get_angle(v,u_vec) > 0 ) and (get_angle(v,u_vec) < np.pi)):
        sgn_u = -1
    u0_helio = np.linalg.norm(u_vec)*sgn_u

    return(t0_helio, tE_helio, u0_helio, piEN_helio, piEE_helio)
