"""
! this calculates the value of Eq. 18 from V.Batista et al. MOA-2009-BLG-387 ! ! 1. expected galactic density (
thick disk+Bulge) ! 2. likelihood of the given proper motion (for a thick disk lens) ! 3. mass function ! 4. jacobian
 || d(Dl, M, _mu_) / d(tE, _mu_, _piE_) || ! ! weights of every link should be muliplied by the value of
 'get_gal_jac' ! to account for priors and trasformation from physical to ulensing parameters ! thetaE ----> _mu_ !
 2021.02. ADDED mu_S from Gaia, using for computing expected mu_rel ! 2021.02. Bulge prior added, now testing both
 Bulge and Disk ! 2021.03.08: velocities of the Sun and galactic coords computed inside the procedure. ! 2021.03.08:
 switch to Bulge added ! 2021.11.02: KK - added better determination of v_d_1 and v_d_2, following PMroz's comments (
 Mroz et al 2019, Reid et al 2009) ! 2021.11.04: KK - changed velocity determination to be in the same frame as in
 Mroz et al 2019, checked on 05.11.2021 with P Mroz, and the results are correct :))
"""

'''
KK notes: In Batista et al. 2011: bulge spans form 5kpc<=d<=11kpc
          and disk spans from 0<=d<=7kpc. To be incorporated???
'''

import numpy as np
from packages.const import DISTANCE_GC, V_ROT, U_SUN_LSR, V_SUN_LSR, W_SUN_LSR, RA_GAL_N, DEC_GAL_N, L_EQ_N

'''
========================================================================
dens_prob_bulge(dist, gall, galb)
========================================================================
Returns probability of finding star in a given direction in the Galactic
disc.
Comes from Batista et al. 2011
------------------------------------------------------------------------
Input:
dist -- distance to the object
gl, gb -- galactic longtitude and latitude
'''
def dens_prob_disc(dist, gall, galb):
    xg = DISTANCE_GC - dist * np.cos(gall) * np.cos(galb)
    yg = dist * np.sin(gall) * np.cos(galb)
    zg = dist * np.sin(galb)
    rg = np.sqrt(xg ** 2 + yg ** 2)

    # thick+thin disk scale height and scale length
    h1 = 0.156  # kpc
    h2 = 0.439  # kpc
    Beta = 0.381
    H = 2.75  # kpc
    # exponential disk model from Batista (Zheng model)
    densprob = 1.07 * np.exp(-rg / H) * ((1 - Beta) * np.exp(-abs(zg) / h1) + Beta * np.exp(-abs(zg) / h2))

    return densprob

'''
========================================================================
dens_prob_bulge(dist, gall, galb)
========================================================================
Returns probability of finding star in a given direction in the Galactic
bulge.
Comes from Batista et al. 2011
------------------------------------------------------------------------
Input:
dist -- distance to the object
gl, gb -- galactic longtitude and latitude
'''
def dens_prob_bulge(dist, gall, galb):
    xg = DISTANCE_GC - dist * np.cos(gall) * np.cos(galb)
    yg = dist * np.sin(gall) * np.cos(galb)
    zg = dist * np.sin(galb)
    rg = np.sqrt(xg ** 2 + yg ** 2)

    # These values come from Han&Gould 2003.
    x0 = 1.58
    y0 = 0.62
    z0 = 0.43
    xp, yp = rotate(xg, yg, np.deg2rad(20))
    zp = zg
    rs4 = ((xp / x0) ** 2 + (yp / y0) ** 2) ** 2 + (zp / z0) ** 4
    rr = np.sqrt(xp ** 2 + yp ** 2)
    densprob = 1.23 * np.exp(-0.5 * np.sqrt(rs4))  # CORRECT

    # CHANGED
    if rr > 2.4:
        densprob *= np.exp(-0.5 * ((rr - 2.4) / 0.5) ** 2)

    return densprob


# input: murel_helio (random), te_helio, piEN_helio, piEE_helio (converted to the heliocentric reference frame), all heliocentric, equatorial
# mura, mudec - original Gaia heliocentric, equatorial, will be converted to geo, gal
'''
========================================================================
get_gal_jac_gaia(mass, dist_lens, dist_source, piE, ra, dec, pien, piee, murel, te, gl, gb, masspower, mu_ra,
                 mu_dec, sig_mu_ra, sig_mu_dec, pm_corr, ds_weight)
========================================================================
Returns probability of finding star with given parameters in the Galaxy.
------------------------------------------------------------------------
Input:
mass -- mass of the lens
dist_lens -- distance to the lens
dist_source -- distance to the source
piE -- microlensing parallax of the analyzed model
ra, dec -- right ascention and declitnation of the event
pien, piee -- north and east componenets of microlensing parallax pie of the analysed model
murel -- relative proper motion betwen the source and lens
te -- Einstein timescale of for the analysed best model
gl, gb -- galactic longtitude and latitude
masspower -- power of the assumed mass-spectrum (Mlens propto M^(-masspower)
mu_ra, mu_dec -- proper motions in ra and dec, from Gaia catalog
sig_mu_ra, sig_mu_dec -- proper motions errors (from Gaia catalog)
pm_corr -- correlation between  proper motions in ra and dec (from Gaia catalog)
ds_weight -- boolean argument, if True, the properties of the source are also weighted by the galactic model
'''
def get_gal_jac_gaia(mass, dist_lens, dist_source, piE, ra, dec, pien, piee, murel, te, gl, gb, masspower, mu_ra,
                     mu_dec, sig_mu_ra, sig_mu_dec, pm_corr, ds_weight):
    VRo = V_ROT  # in km/s, from Mroz+ 2019, ,odel 2 without prior, close to IAU value of 220
    Ro = DISTANCE_GC  # in kpc, distance to GC from GRAVITY
    U_sun, V_sun, W_sun = U_SUN_LSR, V_SUN_LSR, W_SUN_LSR #from Schoenberg 2010
    # from Mroz et al. 2019 (for model 2, with prior)

    # lens and source distances
    pil = 1 / dist_lens

    def V(r):  # for lens, radial velocity depends on the distance, from Mroz et al 2019 (model 2)
        return VRo - 1.34 * (r - Ro)  # note that significant changes only very close to GC or very far from GC

    gall = np.deg2rad(gl)
    galb = np.deg2rad(gb)

    # proper mootions in (l, b) for the source
    pm_l, pm_b, pm_l_err, pm_b_err = eqPMtogalPM(mu_ra, mu_dec, sig_mu_ra, sig_mu_dec, pm_corr, ra, dec, gl, gb,)
    
    mu_sl = pm_l
    mu_sb = pm_b
    sig_mu_sl = pm_l_err
    sig_mu_sb = pm_b_err

    # calculating proper motions for the lens
    # two options, the lens is either in the bulge or in the disk
    # calculating velocities for both
    # based on PMroz comments, Mroz et al 2019 + Reid et al 2009
    # calculating Dp and Rp to get beta
    Dp = dist_lens * np.cos(galb)
    Rp = np.sqrt(Ro ** 2 + Dp ** 2 - (2. * Dp * Ro * np.cos(gall)))
    sinbeta = Dp * np.sin(gall) / Rp
    cosbeta = (Ro - Dp * np.cos(gall)) / Rp
    beta = np.arctan2(sinbeta, cosbeta)  # angle Sun-galactic center-lens, check figure in Reid et al 2009
    R = np.sqrt(Rp ** 2 + (dist_lens * np.sin(galb)) ** 2)  # distance from GC to lens
    # We are correcting the movement of the lens for the movement of the Sun and Earth later on, so I do not subtract
    U_1 = V(R) * np.sin(beta) - U_sun
    V_1 = V(R) * np.cos(beta) - VRo - V_sun
    W_1 = (-1.) * W_sun

    # if the lens is in the disk
    # expected thick disk velocities
    vd_1 = W_1 * np.cos(galb) - (U_1 * np.cos(gall) + V_1 * np.sin(gall)) * np.sin(galb)  # galb
    vd_2 = V_1 * np.cos(gall) - U_1 * np.sin(gall)  # as in Mroz #gall
    # expected dispersion in disk velocities
    svd_1 = 20.
    svd_2 = 30.  # as in Mroz

    # expected bulge source velocities
    # gal l and gal b close to 0 => beta = 0
    vb_1 = (-1.) * W_sun  # 0. #galb
    vb_2 = (-1) * V_sun  # vrot canceles out #as in Mroz #gall
    # expected dispersion in disk velocities, as in Mroz
    svb_1 = 100.
    svb_2 = 100.  # bulge motions are quite random

    # angle between North Pole in eqatorial coordinates, lens and
    # Galactic north pole; used to find north and east galactic components 
    # of murel (check if this is really needed????)
    northPA = calculateNorthPA(ra, dec, gl, gb)
    # northPA = np.deg2rad(60.) # old, wrong value, left, becasue new procedure has not yet been properly teste

    # now the proper motions should be in the same frame
    # relative proper motion we expect (difference of geocentric PMs) in the disk
    mu_exp_1 = (vd_1 * pil / 4.74) - mu_sb  # in AU/yr/kpc = mas/yr , gal b
    mu_exp_2 = (vd_2 * pil / 4.74) - mu_sl  # in AU/yr/kpc = mas/yr , gal l

    # relative proper motion we expect in the bulge
    mu_exp_b1 = (vb_1 * pil / 4.74) - mu_sb  # in AU/yr/kpc = mas/yr , gal b
    mu_exp_b2 = (vb_2 * pil / 4.74) - mu_sl  # in AU/yr/kpc = mas/yr , gal l

    # relative proper motion dispersion we expect in the disk
    smu_exp_1 = np.sqrt((svd_1 * pil / 4.74) ** 2 + sig_mu_sb ** 2)  # in mas/yr, in gal b
    smu_exp_2 = np.sqrt((svd_2 * pil / 4.74) ** 2 + sig_mu_sl ** 2)  # in mas/yr, in gal l
    # relative proper motion dispersion we expect in the bulge
    smu_exp_b1 = np.sqrt((svb_1 * pil / 4.74) ** 2 + sig_mu_sb ** 2)  # in mas/yr, in gal b
    smu_exp_b2 = np.sqrt((svb_2 * pil / 4.74) ** 2 + sig_mu_sl ** 2)  # in mas/yr, in gal l

    # relative proper motion we see
    mu_len = murel
    # spliting to N and E components along piE vector (geo)
    mu_l_1 = mu_len * (pien * np.cos(northPA) - piee * np.sin(northPA)) / piE  # north galactic component in mas/yr
    mu_l_2 = mu_len * (pien * np.sin(northPA) + piee * np.cos(northPA)) / piE  # east  galactic component in mas/yr

    # probability of given proper motion, bulge and disk added
    fmu_b1 = np.exp(-(mu_exp_b1 - mu_l_1) ** 2 / (2. * smu_exp_b1 ** 2)) / smu_exp_b1  # prob bulge
    fmu_b2 = np.exp(-(mu_exp_b2 - mu_l_2) ** 2 / (2. * smu_exp_b2 ** 2)) / smu_exp_b2  #
    fmu_d1 = np.exp(-(mu_exp_1 - mu_l_1) ** 2 / (2. * smu_exp_1 ** 2)) / smu_exp_1  # prob disk
    fmu_d2 = np.exp(-(mu_exp_2 - mu_l_2) ** 2 / (2. * smu_exp_2 ** 2)) / smu_exp_2  #

    # --------------------------------------------
    # very simple mass function weighting assumed here
    mass_function = 1. / (mass ** masspower)
    # --------------------------------------------
    # TOTAL + JACOBIAN
    czlon = (mass_function * mass) * (dist_lens ** 4) * (
            murel ** 4) * te / piE  # *te ze zmiany zmiennej w jakobianie (thetaE na mu)
    # --------------------------------------------
    # DENSITY #edited to match Mroz 2021 notes
    densprob_d = dens_prob_disc(dist_lens, gall, galb)
    gal_jac_d = 1.e1 * densprob_d * fmu_d1 * fmu_d2 * czlon
    # DENS BULGE + NORMALIZATION bulge-disk
    densprob_b = dens_prob_bulge(dist_lens, gall, galb)
    gal_jac_b = 1.e1 * densprob_b * fmu_b1 * fmu_b2 * czlon
    gal_jac = max(gal_jac_b, gal_jac_d)
    if ds_weight:
        densprob_d_source = dens_prob_disc(dist_source, gall, galb)
        densprob_b_source = dens_prob_bulge(dist_source, gall, galb)
        gal_jac *= max(densprob_d_source, densprob_b_source)

    return gal_jac

'''
========================================================================
rotate (x_old, y_old, angle)
========================================================================
Returns x and y rotated in clockwise motion by an angle.
------------------------------------------------------------------------
Input:
x_old, y_old -- coordinates of an object that we want to rotate
angle -- angle by which we want to rotate our object
'''
def rotate(x_old, y_old, angle):
    x_new = x_old * np.cos(angle) + y_old * np.sin(angle)
    y_new = -x_old * np.sin(angle) + y_old * np.cos(angle)
    return x_new, y_new
    
'''
========================================================================
eqPMtogalPM(mu_ra, mu_dec, ra, dec, l, b)
========================================================================
Returns proper motions in galactic coordinate system.
From Poleski 2013.
------------------------------------------------------------------------
Input:
mu_ra, mu_dec -- proper motion in equatorial coordinates
ra, dec -- equatorial coordinates of a source with given proper motions
l, b -- galactic coordinates of a source with given proper motions
pm_corr -- correlation betwen mu_ra and mu_dec, from Gaia catalogue
'''
def eqPMtogalPM(mu_ra, mu_dec, sig_mu_ra, sig_mu_dec, pm_corr, ra, dec, l, b,):
    # converting proper motions od the source to galactocentric frame
    # following Reid et al. 2009 and Mroz et al. 2019
    # 1) convert proper motions to mu_l, mu_b, from
    # http://www.astrouw.edu.pl/ogle/ogle4/ROTATION_CURVE/construct_rotation_curve.py
    # by P. Mroz
    # formulae come from Poleski 2013
    pm_ra, pm_ra_err = mu_ra, sig_mu_ra
    pm_dec, pm_dec_err = mu_dec, sig_mu_dec
    ra_G = np.deg2rad(RA_GAL_N)  # like in Poleski 2013
    dec_G = np.deg2rad(DEC_GAL_N)  # like in Poleski 2013
    # ra, dec in degrees, converting
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    C1 = np.sin(dec_G) * np.cos(dec) - np.cos(dec_G) * np.sin(dec) * np.cos(ra - ra_G)
    C2 = np.cos(dec_G) * np.sin(ra - ra_G)
    norm = np.sqrt(C1 * C1 + C2 * C2)
    C1 /= norm
    C2 /= norm
    pm_l = C1 * pm_ra + C2 * pm_dec
    pm_b = C1 * pm_dec - C2 * pm_ra
    # Transforming error bars of proper motions
    J1 = np.array([[C1, C2], [-C2, C1]])
    J2 = np.transpose(J1)
    CORR = np.array([[pm_ra_err ** 2, pm_corr * pm_ra_err * pm_dec_err],
                     [pm_corr * pm_ra_err * pm_dec_err, pm_dec_err ** 2]])
    COV = np.dot(np.dot(J1, CORR), J2)
    pm_l_err = np.sqrt(COV[0, 0])
    pm_b_err = np.sqrt(COV[1, 1])
    
    return pm_l, pm_b, pm_l_err, pm_b_err


'''
========================================================================
calculateNorthPA(ra, dec, gl, gb)
========================================================================
Returns the angle between equatorial north pole, lens and
Galactic north pole.
RETURNED ANGLE IS IN RADIANS!!!!
------------------------------------------------------------------------
Input:
ra, dec -- equatorial coordinates of the lens
l, b -- galactic coordinates of the lens
# '''
def calculateNorthPA(ra, dec, l, b):
    raRad, decRad = np.deg2rad(ra), np.deg2rad(dec)
    lRad, bRad = np.deg2rad(l), np.deg2rad(b)
    sinNPA = np.sin(raRad - np.deg2rad(RA_GAL_N)) * np.cos(np.deg2rad(DEC_GAL_N)) / np.cos(bRad)
    cosNPA = (np.sin(np.deg2rad(DEC_GAL_N)) - (np.sin(bRad) * np.sin(decRad))) / (np.cos(decRad) * np.cos(bRad))
    
    NPA = np.arctan2(sinNPA, cosNPA)
    return NPA
