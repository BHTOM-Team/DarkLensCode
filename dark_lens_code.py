import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from src.gal_jac_dens import *
from src.mass_luminosity import *
from src.geo_helio import get_Earth_pos
import src.const as const
import click
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from wquantiles import quantile
from typing import Tuple
import yaml
from asymmetric_uncertainty import a_u


def get_mass_distance(iteration: int):
    # arguments ordered like in MulensModel output
    t0 = data[iteration][0]
    u0 = data[iteration][1]
    tE = data[iteration][2]
    piEN = data[iteration][3]
    piEE = data[iteration][4]
    mag0 = data[iteration][5]
    flux_fraction = data[iteration][6]

    if (par.get('f_source')):
        f_blend = 1. - flux_fraction
        fs = flux_fraction
    else:
        f_blend = flux_fraction
        fs = 1. - flux_fraction

    if fs < 1:
        mag_blend = mag0 - 2.5 * np.log10(f_blend)
        mag_source = mag0 - 2.5 * np.log10(1. - f_blend)
    else:
        mag_blend = const.MIN_MAG
        mag_source = mag0

    if mag_blend > const.MIN_MAG:
        mag_blend = const.MIN_MAG

    err_source_mag = par.get('source_mag_err') # error from model

    dist_source_good = False
    dist_source = 0
    while(not dist_source_good):
        if par.get('ds_weight'):
            dist_source = np.random.uniform(par.get('ds_median')-par.get('ds_err_neg'),
                                            par.get('ds_median')+par.get('ds_err_pos'))
        else:
            au = a_u(par.get('ds_median'), par.get('ds_err_pos'), par.get('ds_err_neg'))
            ds_min = max(0.1, par.get('ds_median') - 3*par.get('ds_err_neg'))
            ds_max = par.get('ds_median') + 3*par.get('ds_err_pos')
            distances = np.linspace(ds_min, ds_max, par.get('ds_samples'))
            probs = au.pdf(distances)
            dist_source = np.random.choice(distances, p=probs/np.sum(probs))

        ext_source = 0.0
        coeffs = par['extinction_coeffs']
        if(dist_source > 8.):
            for i in range(len(coeffs)):
                # print(coeffs[i], dist_source, i)
                ext_source += coeffs[i] * (8000.) ** i
        else:
            for i in range(len(coeffs)):
                # print(coeffs[i], dist_source, i)
                ext_source += coeffs[i] * (dist_source * 1000.)**i

        # check if distance to the source matches observed mag and theta_star
        theta_star = float(par.get('theta_star'))
        err_theta_star = float(par.get('theta_star_sig'))
        radius_star = 206.265 * theta_star * dist_source / 215.03 # in solar radii
        err_radius_star = 206.265 * err_theta_star * dist_source / 215.03
        obs_source_mag = -100.
        err_obs_source_mag = 0.
        if (par.get('source_star') == 'dwarf'):
            obs_source_mag, err_obs_source_mag = get_dwarf_obs_mag(par.get('rad_fun'),
                                               par.get('t_eff_fun'),
                                               radius_star,
                                               err_radius_star,
                                               dist_source, 'G',
                                               ext_source,
                                               par.get('rad_mag_min'), par.get('rad_mag_max'),
                                               par.get('t_eff_mag_min'), par.get('t_eff_mag_max'))
            # print(obs_source_mag, err_obs_source_mag)
        elif (par.get('source_star') == 'giant'):
            obs_source_mag, err_obs_source_mag = get_giant_obs_mag_g(radius_star, err_radius_star, dist_source, ext_source)

        # print("In loop: dist= %.2f kpc, mag_source model = %.2f, mag_source radius = %.2f, diff= %.2f, error= %.2f"%
        #       (dist_source, mag_source, obs_source_mag, np.abs(mag_source - obs_source_mag), (err_source_mag + err_obs_source_mag)))
        # print(err_obs_source_mag)
        if (np.abs(mag_source - obs_source_mag) < (err_source_mag + err_obs_source_mag)):
            dist_source_good = True

    # print("Out of loop: ", dist_source)

    mass_in_bounds = False
    mass_lens = 0
    while(not mass_in_bounds):
        mu_rel = np.random.uniform(0, 30)
        piE = np.sqrt(piEN ** 2 + piEE ** 2)
        mass_lens = mu_rel * ((tE / const.YEAR) / piE) / const.KAPPA
        dist_lens = 1 / (mu_rel * (tE / const.YEAR) * piE + 1 / dist_source)
        if ((mass_lens < par.get('mass_break')[-1])):
            mass_in_bounds = True


    # conversion from geocentric to heliocentric reference frame
    v_Earth_perp = par['v_Earth_perp']
    piE_geo = np.array([piEN, piEE])
    tE_geo = tE
    tE_helio_inverse = (1 / tE_geo) * (piE_geo / piE) + v_Earth_perp * piE
    piE_helio = piE * tE_helio_inverse / np.linalg.norm(tE_helio_inverse)
    piEN_helio, piEE_helio = piE_helio[0], piE_helio[1]
    tE_helio = 1 / np.linalg.norm(tE_helio_inverse)

    w_gal = get_gal_jac_gaia(mass_lens, dist_lens, dist_source,
                             piE,
                             par.get('alpha'), par.get('delta'),
                             piEN_helio, piEE_helio,
                             mu_rel, tE_helio,
                             par.get('gal_l'), par.get('gal_b'),
                             par.get('mass_pows'), par.get('mass_break'),
                             par.get('mu_ra'), par.get('mu_dec'),
                             par.get('mu_ra_sig'), par.get('mu_dec_sig'),
                             par.get('mu_ra_dec_corr'),
                             par.get('ds_weight'),  par.get('mus_weight'))

    #Added modification to be compatible with Kruszynska et al.2024
    # If f_source is set to True -> the flux_fraction in the ulens PDF is the fraction
    # of total light coming from the source (f_s). Otherwise it's the blends fraction (f_b).

    extinction = 0
    coeffs = par['extinction_coeffs']
    for i in range(len(coeffs)):
        extinction += coeffs[i] * (dist_lens * 1000.)**i

    mag_lens_w_extinction = get_ms_mag(par.get('fun'), mass_lens, dist_lens, par.get('filter'), extinction,
                                       par.get('mag_min'), par.get('mag_max'))
    # mag_lens_0_extinction = get_ms_mag(par.get('fun'), mass_lens, dist_lens, par.get('filter'), 0.0,
    #                                    par.get('mag_min'), par.get('mag_max'))
    # if fs < 0:
    #     mag_source = const.MIN_MAG
    # else:
    #     mag_source = mag0 - 2.5 * np.log10(fs)

    thetaE = tE * mu_rel / 365.25

    # return [w_gal, mass_lens, dist_lens, mag_blend, mag_lens_w_extinction, mag_lens_0_extinction, mag_source, t0, tE,
    #         u0, piEN, piEE, mag0, fs, mu_rel, thetaE]
    return [w_gal, mass_lens, dist_lens, mag_blend, mag_lens_w_extinction, mag_source, t0, tE,
            u0, piEN, piEE, mag0, fs, mu_rel, thetaE]


def mp_mass_distance(n_iter: int):
    results = []
    print('---- STARTING MULTIPROCESSING ----')
    start_time = time.time()
    set_start_method('fork')
    iterations = np.random.randint(len(data), size=n_iter)
    with Pool() as pool:
        with tqdm(total=n_iter) as pbar:
            for i in pool.imap_unordered(get_mass_distance, iterations):
                results.append(i)
                pbar.update()
    print(f'Multiprocessing took {time.time() - start_time:.2f} seconds\n')
    return results


def parse_input_parameters(parameters: dict, options: dict):
    global par
    par = parameters.copy()
    try:
        par['masspower'] = options['masspower']
    except KeyError:
        par['masspower'] = '-1.75, 1000.'
    try:
        par['filter'] = options['filter']
    except KeyError:
        par['filter'] = 'V'
    try:
        par['ds_weight'] = options['ds_weight']
    except KeyError:
        par['ds_weight'] = False
    try:
        par['mus_weight'] = options['mus_weight']
    except KeyError:
        par['mus_weight'] = False
    try:
        par['f_source'] = options['f_source']
    except KeyError:
        par['f_source'] = True
    try:
        par['n_iter'] = int(options['n_iter'])
    except KeyError:
        par['n_iter'] = int(1e6)
    try:
        par['ds_samples'] = int(options['ds_samples'])
    except KeyError:
        par['ds_samples'] = 10000

    try:
        print('INPUT PARAMETERS: ')
        print('alpha = ', par['alpha'])
        print('delta = ', par['delta'])
        print('t0par = ', par['t0par'])
        print('extinction = ', par['extinction'])
        print('source_mag_err = ', par['source_mag_err'])
        print('ds_median = ', par['ds_median'])
        print('ds_err_pos = ', par['ds_err_pos'])
        print('ds_err_neg = ', par['ds_err_neg'])
        print('ds_samples = ', par['ds_samples'])
        print('source_mag_err = ', par['source_mag_err'])
        print('source_star = ', par['source_star'])
        print('theta_star = ', par['theta_star'])
        print('theta_star_sig = ', par['theta_star_sig'])
        print('mu_ra = ', par['mu_ra'])
        print('mu_ra_sig = ', par['mu_ra_sig'])
        print('mu_dec = ', par['mu_dec'])
        print('mu_dec_sig = ', par['mu_dec_sig'])
        print('mu_ra_dec_corr = ', par['mu_ra_dec_corr'])
        print('filter = ', par['filter'])
        print('masspower = ', par['masspower'])
        print('DS_weight = ', par['ds_weight'])
        print('n_iter = ', par['n_iter'])
        print('\n')
    except KeyError:
        print('Missing one or more parameters. Check YAML file.')
        exit(1)


# def take_results_parameters(results_t: list):
#     res_par = {'weights': results_t[0] / np.sum(results_t[0]), 'ML': results_t[1], 'DL': results_t[2],
#                'mag_blend': results_t[3], 'mag_lens_w_extinction': results_t[4], 'mag_lens_0_extinction': results_t[5],
#                'mag_source': results_t[6], 't0': results_t[7], 'tE': results_t[8], 'u0': results_t[9],
#                'piEN': results_t[10], 'piEE': results_t[11], 'mag0': results_t[12], 'fs': results_t[13],
#                'theta_E': results_t[14]}
#     return res_par

def take_results_parameters(results_t: list):
    res_par = {'weights': results_t[0] / np.sum(results_t[0]), 'ML': results_t[1], 'DL': results_t[2],
               'mag_blend': results_t[3], 'mag_lens_w_extinction': results_t[4],
               'mag_source': results_t[5], 't0': results_t[6], 'tE': results_t[7], 'u0': results_t[8],
               'piEN': results_t[9], 'piEE': results_t[10], 'mag0': results_t[11], 'fs': results_t[12],
               'theta_E': results_t[13]}
    return res_par


def save_samples(filename: str, results: list):
    print('--- SAVING SAMPLES ---')
    with open(filename, 'wb') as handle:
        np.save(handle, results)
    print(f'Samples saved: {filename}\n')


def get_probability(res_par: dict) -> Tuple[float, float]:
    # ZIP = np.array([res_par.get('mag_blend'), res_par.get('ML'), res_par.get('weights'),
    #                 res_par.get('DL'), res_par.get('mag_lens_0_extinction'), res_par.get('mag_lens_w_extinction')]).T
    ZIP = np.array([res_par.get('mag_blend'), res_par.get('ML'), res_par.get('weights'),
                    res_par.get('DL'), res_par.get('mag_lens_w_extinction')]).T
    # remn_lower_limit = ZIP[ZIP[:, 5] < ZIP[:, 0]]
    # nonrem_lower_limit = ZIP[ZIP[:, 5] >= ZIP[:, 0]]
    remn_upper_limit = ZIP[ZIP[:, 4] < ZIP[:, 0]]
    nonrem_upper_limit = ZIP[ZIP[:, 4] >= ZIP[:, 0]]

    # remnant_lower_limit = np.sum(remn_lower_limit[:, 2])
    # nonremnant_lower_limit = np.sum(nonrem_lower_limit[:, 2])

    remnant_upper_limit = np.sum(remn_upper_limit[:, 2])
    nonremnant_upper_limit = np.sum(nonrem_upper_limit[:, 2])

    # prob_lower_limit = 100 * remnant_lower_limit / (nonremnant_lower_limit + remnant_lower_limit)
    prob_upper_limit = 100 * remnant_upper_limit / (nonremnant_upper_limit + remnant_upper_limit)
    # print(f'Dark lens probability =  [{prob_lower_limit:.2f}%, {prob_upper_limit:.2f}%]')
    print(f'Dark lens probability =  {prob_upper_limit:.2f}%')
    # return prob_lower_limit, prob_upper_limit
    return prob_upper_limit


def save_results(filename: str, res_par: dict):
    median_ml = quantile(res_par.get('ML'), res_par.get('weights'), const.MEDIAN)
    plus_ml = quantile(res_par.get('ML'), res_par.get('weights'), const.SIGMA_PLUS)
    minus_ml = quantile(res_par.get('ML'), res_par.get('weights'), const.SIGMA_MINUS)
    median_dl = quantile(res_par.get('DL'), res_par.get('weights'), const.MEDIAN)
    plus_dl = quantile(res_par.get('DL'), res_par.get('weights'), const.SIGMA_PLUS)
    minus_dl = quantile(res_par.get('DL'), res_par.get('weights'), const.SIGMA_MINUS)
    median_thetaE = quantile(res_par.get('theta_E'), res_par.get('weights'), const.MEDIAN)
    plus_thetaE = quantile(res_par.get('theta_E'), res_par.get('weights'), const.SIGMA_PLUS)
    minus_thetaE = quantile(res_par.get('theta_E'), res_par.get('weights'), const.SIGMA_MINUS)

    print('--- CALCULATING RESULTS ---')
    # prob_lower_limit, prob_upper_limit = get_probability(res_par)
    prob_upper_limit = get_probability(res_par)
    print(f'Mass of the lens: {median_ml:.2f} +{plus_ml - median_ml:.2f} -{median_ml - minus_ml:.2f} [Solar masses]')
    print(f'Distance do the lens: {median_dl:.2f} +{plus_dl - median_dl:.2f} -{median_dl - minus_dl:.2f} [kpc]')
    print(f'Theta_E: {median_thetaE:.2f} +{plus_thetaE - median_thetaE:.2f} -{median_thetaE - minus_thetaE :.2f} [mas]\n')

    print('--- SAVING RESULTS ---')
    header = 'ML_median ML_sigma+ ML_sigma- DL_median DL_sigma+ DL_sigma- DarkLensProb_lower_limit DarkLensProb_upper_limit'
    mass_sigma_plus = plus_ml - median_ml
    mass_sigma_minus = median_ml - minus_ml
    mass = f'{median_ml:.5f} +{mass_sigma_plus:.5f} -{mass_sigma_minus:.5f}'
    dist_sigma_plus = plus_dl - median_dl
    dist_sigma_minus = median_dl - minus_dl
    dist = f'{median_dl:.5f} +{dist_sigma_plus:.5f} -{dist_sigma_minus:.5f}'
    thetaE_sigma_plus = plus_thetaE - median_thetaE
    thetaE_sigma_minus = median_thetaE - minus_thetaE
    thetaE = f'{median_thetaE:.5f} +{thetaE_sigma_plus:.5f} -{thetaE_sigma_minus:.5f}'
    # res = f'mass: {mass}\ndist: {dist}\ntheta_E: {thetaE}\nprob: {prob_lower_limit:.5f} {prob_upper_limit:.5f}'
    res = f'mass: {mass}\ndist: {dist}\ntheta_E: {thetaE}\nprob: {prob_upper_limit:.5f}'
    print(f'Results saved: {filename}\n')
    with open(filename, 'w') as f:
        f.write(res)


def parse_yaml_file(settings: dict):
    try:
        parameters = settings['parameters']
    except KeyError:
        print('Missing parameters. Check YAML file.')
        exit(1)
    try:
        input_filename = settings['input']['samples']['file_name']
    except KeyError:
        print('Missing input file name. Check YAML file.')
        exit(1)
    try:
        samples_filename = settings['output']['samples']['samples_filename']
    except KeyError:
        print('Missing output file name. Check YAML file.')
        exit(1)
    try:
        results_filename = settings['output']['samples']['results_filename']
    except KeyError:
        print('Missing output file name. Check YAML file.')
        exit(1)
    try:
        options = settings['options']
    except KeyError:
        options = {}

    return parameters, input_filename, samples_filename, results_filename, options


def load_input_samples(input_filename: str):
    print('--- READING INPUT SAMPLES ---')
    try:
        global data
        data = []
        chains = np.load(input_filename, allow_pickle=True)
        for params in chains:
            data.append(params)
    except FileNotFoundError:
        print(f'No input file {input_filename}: check the path provided in the YAML file.')
        exit(1)
    except ValueError:
        print(f'Badly formatted input file {input_filename}. Provide an ASCII filename.')
        exit(1)
    except Exception:
        print(f'Something went wrong with input samples {input_filename}.')
        exit(1)


@click.command()
@click.argument('yaml-file', type=click.Path(exists=True))
def main(yaml_file: str):

    with open(yaml_file, 'r') as input_yaml:
        settings = yaml.safe_load(input_yaml)
    # Parsing YAML file
    parameters, input_filename, samples_filename, results_filename, options = parse_yaml_file(settings)
    parse_input_parameters(parameters, options)
    # Reading input data 
    load_input_samples(input_filename)
    # Converting to galactic coordinates
    eq2gal = SkyCoord(ra=par['alpha'] * u.degree, dec=par['delta'] * u.degree,
                      frame='icrs')
    par['gal_l'] = eq2gal.galactic.l.degree
    par['gal_b'] = eq2gal.galactic.b.degree
    # Initializing mass-luminosity function
    par['fun'], par['mag_min'], par['mag_max'] = initialize_interp(par['filter'])
    # Initializing mass-function
    par['mass_pows'], par['mass_break'] = initialize_mass_fun(par['masspower'])
    # Initializing radius-magnitude function for dwarf stars
    par['rad_fun'], par['rad_mag_min'], par['rad_mag_max'] = initialize_interp_radius_dwarf(par['filter'])
    # Initializing radius-effective temp function for dwarf stars
    par['t_eff_fun'], par['t_eff_mag_min'], par['t_eff_mag_max'] = initialize_interp_teff_dwarf(par['filter'])
    # Initialize complex extinction
    par['extinction_coeffs'] = initialize_extinction(par.get('extinction'))
    # Getting relative Earth velocity at the reference time - for further geo-helio conversion
    par['x_Earth_perp'], par['v_Earth_perp'] = get_Earth_pos(par.get('t0par'), par.get('alpha'), par.get('delta'))
    # Multiprocessing
    results = mp_mass_distance(par['n_iter'])
    # Saving samples and results
    res_par = take_results_parameters(np.array(results).T)
    save_samples(samples_filename, results)
    save_results(results_filename, res_par)


if __name__ == '__main__':
    main()
