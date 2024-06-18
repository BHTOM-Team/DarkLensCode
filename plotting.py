import os
from typing import List, Dict, Tuple, Optional, Callable
from warnings import filterwarnings

import click
import yaml
import corner
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.transforms import Bbox
from scipy.optimize import fmin
from wquantiles import quantile
import scipy.stats as ss
import pathlib
from scipy import optimize

from src import const
from dark_lens_code import take_results_parameters
from turbo_colormap import turbo_colormap_data

filterwarnings('ignore', 'divide by zero encountered in log10')

SIGMA_LEVELS: List[float] = [0.382, 0.6827,
                             0.866, 0.9545, 0.987, 0.9973, 0.999]
SIGMA1: float = 0.6827
SIGMA3: float = 0.9973

PEACOCK_COLORS: List[str] = ["#081582", "#3cadee",
                             "#9e95c6", "#11fcff", "#73dfa8", "#fffd7d"]

TURBO_INDICES = np.linspace(0, 255, 6)
TURBO: List[List[float]] = [turbo_colormap_data[int(i)] for i in TURBO_INDICES]
HIST1D_COLOR = turbo_colormap_data[-2]


def generate_plot_name(plot_filename: str, plot_suffix: str) -> str:
    file_extension: str = ''.join(pathlib.Path(plot_filename).suffixes)
    return plot_filename.replace(file_extension, f'_{plot_suffix}{file_extension}')


# --- LIKELIHOOD AND CONFIDENCE --- #

def integrate_for_pdf(value: float,
                      values: np.array) -> float:
    """

    Args:
        value: Threshold PDF value
        values: All PDF values

    Returns:
        Fraction of samples with PDF>=threshold
    """
    return np.sum([v for v in values
                   if v >= value]) / np.sum(values)


def get_pdf_for_confidence(confidence: float,
                           pdf_values: np.array) -> float:
    """

    Args:
        confidence: Desired confidence level
        pdf_values: All PDF values for samples

    Returns:
        Threshold PDF corresponding to the passed confidence
    """
    return fmin(lambda pdfval: np.abs(integrate_for_pdf(pdfval, pdf_values) - confidence),
                np.max(pdf_values) * .75, disp=False)[0]


def get_x_for_pdf(pdf: Callable[[float], float],
                  pdf_value: float,
                  samples: np.array) -> float:
    """

    Args:
        pdf: PDF function
        pdf_value: desired PDF value
        samples: sample values

    Returns:
        value of the sample with passed PDF value
    """
    return fmin(lambda x: np.abs(pdf(x) - pdf_value), np.median(samples), disp=False)[0]


def integrate_for_likelihood(log_likelihood: float,
                             log_likelihood_values: np.array) -> float:
    """
    Args:
        log_likelihood (float): Threshold log likelihood value
        log_likelihood_values (List[float]): Log likelihood values for all samples

    Returns:
        float: Fraction of samples below the log likelihood threshold
    """
    values = log_likelihood_values.flatten()
    values = values[np.isfinite(values)]

    if log_likelihood >= 0:
        return -np.inf

    return np.sum([v for v in values if v <= log_likelihood]) / np.sum(values)


def get_confidence_interval(confidence: float,
                            values: np.array) -> float:
    """
    Args:
        confidence (float): Confidence level (the fraction of samples below some threshold log likelihood)
        values (np.array): Samples

    Returns:
        float: Log likelihood value corresponding to the confidence level
    """
    return fmin(lambda x: np.abs(integrate_for_likelihood(x, values) - confidence),
                np.median(values[np.isfinite(values)]), disp=False)[0]


# --- INPUT PARSING --- #

def read_result_parameters(filename: str) -> Dict[str, np.array]:
    """
    Parse samples (result of darklens run) into dictionary.

    Args:
        filename: samples filename

    Returns:
        dictionary of sample parameters
    """
    try:
        samples: np.array = np.load(filename)
        res_par: Dict[str, np.array] = take_results_parameters(samples.T)

        return res_par
    except FileNotFoundError:
        print(f'File {filename} not found!')
        exit(2)


def parse_yaml_file(settings: dict) -> Tuple[dict, str, str]:
    """
    Parse the settings and parameters contained in the YAML file.
    Args:
        settings: YAML file (use yaml.safe_load for that)

    Returns:
        parameters (as a dictionary), samples_filename, plot_filename

    """
    try:
        parameters = settings['parameters']
    except KeyError:
        print('Missing parameters. Check YAML file.')
        exit(1)
    try:
        samples_filename = settings['output']['samples']['samples_filename']
    except KeyError:
        print('Missing output file name. Check YAML file.')
        exit(1)

    try:
        plot_filename = settings['output']['plot']['plot_filename']
    except KeyError:
        print('Missing plot file name. Check YAML file.')
        exit(1)

    return parameters, samples_filename, plot_filename


# --- HISTOGRAM PLOTS HELPER FUNCTIONS --- #

def __continuous_colorbar(map: plt.axes):
    cbar = plt.colorbar(map, ticks=[1e-5, 1e-4, 1e-3, 1e-2])
    cbar.ax.set_yticklabels(['-5', '-4', '-3', '-2'])
    cbar.set_label("log prob density", labelpad=0)


def __contour_colorbar(map: plt.axes, sigma_values: List[float]):
    cbar = plt.colorbar(map, ticks=sigma_values)
    cbar.ax.set_yticklabels(
        ['$4\sigma$', '$3.5\sigma$', '$3\sigma$', '$2.5\sigma$', '$2\sigma$', '$1.5\sigma$', '$1\sigma$'])
    cbar.set_label("confidence level", labelpad=5)


def __continuous_mass_dist_histogram(histogram: np.array,
                                     x_edges: np.array,
                                     y_edges: np.array,
                                     axis: plt.axes,
                                     include_cbar: bool = True):
    map = axis.pcolormesh(y_edges, x_edges, histogram /
                          np.sum(histogram), cmap='turbo', norm=LogNorm(1e-5, 1e-2))

    if include_cbar:
        __continuous_colorbar(map)


def __contour_mass_distance_histogram(histogram: np.array,
                                      x_edges: np.array,
                                      y_edges: np.array,
                                      axis: plt.axes,
                                      color_blind_sigma_contours: bool = False,
                                      include_cbar: bool = True):
    colors: List[str] = TURBO if color_blind_sigma_contours else PEACOCK_COLORS

    normalized_hist = np.log10(histogram / np.sum(histogram))
    sigma_values: List[float] = [get_confidence_interval(
        sigma, normalized_hist) for sigma in SIGMA_LEVELS]

    map = axis.contourf(y_edges[:-1], x_edges[:-1], normalized_hist,
                        levels=sigma_values, colors=colors, extend='both')

    if include_cbar:
        __contour_colorbar(map, sigma_values)


def __continuous_blend_lens_histogram(histogram: np.array,
                                      x_edges: np.array,
                                      y_edges: np.array,
                                      axis: plt.axes,
                                      include_cbar: bool = True):
    try:
        map = axis.pcolormesh(y_edges, x_edges, histogram / np.sum(histogram),
                              cmap='turbo', norm=LogNorm(1e-5, 1e-2),
                              edgecolors='None', linewidth=0)
    # turbo isn't available in older versions of matplotlib
    except ValueError:
        map = axis.pcolormesh(y_edges, x_edges, histogram / np.sum(histogram),
                              cmap='jet', norm=LogNorm(1e-5, 1e-2),
                              edgecolors='None', linewidth=0)

    if include_cbar:
        __continuous_colorbar(map)


def __contour_blend_lens_histogram(histogram: np.array,
                                   x_edges: np.array,
                                   y_edges: np.array,
                                   axis: plt.axes,
                                   color_blind_sigma_contours: bool = False,
                                   include_cbar: bool = True):
    colors: List[str] = TURBO if color_blind_sigma_contours else PEACOCK_COLORS

    normalized_hist = np.log10(histogram / np.sum(histogram))

    sigma_values: List[float] = [get_confidence_interval(
        sigma, normalized_hist) for sigma in SIGMA_LEVELS]

    map = axis.contourf(y_edges[:-1], x_edges[:-1], normalized_hist,
                        levels=sigma_values, colors=colors, extend='both')

    if include_cbar:
        __contour_colorbar(map, sigma_values)


def __plot_extinction_lines(extinction: float, axis: plt.axes):
    axis.plot(np.linspace(-20, 30, 100), np.linspace(-20, 30, 100), 'w-')
    # axis.plot(np.linspace(-20, 30, 100) - extinction,
    #           np.linspace(-20, 30, 100), 'w--')
    axis.fill_between(np.linspace(-20, 30, 100), np.linspace(-20, 30, 100), 30, alpha=0.1, facecolor='k',
                      edgecolor='None')


# --- HISTOGRAM PLOTS --- #


def __filter_for_blend_eq_lens(results: Dict[str, float],
                               property_name: str) -> np.array:
    mag_blend: np.array = results.get("mag_blend")
    # mag_lens_0_extinction: np.array = results.get("mag_lens_0_extinction")
    mag_lens_w_extinction: np.array = results.get("mag_lens_w_extinction")
    # blend_lens_mask: np.array = np.argwhere(
    #     (mag_blend <= mag_lens_w_extinction) & (mag_blend >= mag_lens_0_extinction)).flatten()
    blend_lens_mask: np.array = np.argwhere(
        (mag_blend <= mag_lens_w_extinction)).flatten()

    return results.get(property_name)[blend_lens_mask]


def mass_distance_plot(parameters: Dict[str, float],
                       results: Dict[str, float],
                       peacock_sigma_contours: bool = False,
                       turbo_sigma_contours: bool = False,
                       blend_eq_lens: bool = False,
                       plotname: Optional[str] = None,
                       histogram_bins: int = 100,
                       figsize: Tuple[int, int] = (8, 5),
                       axis: Optional[plt.axes] = None,
                       x_limits: Tuple[float, float] = None,
                       y_limits: Tuple[float, float] = None,
                       include_cbar: bool = True):
    """
    Plot the mass-distance plot
    Args:
        parameters: parameters parsed from the yaml file
        results: results loaded from samples
        peacock_sigma_contours: plot filled sigma contours using peacock palette
        turbo_sigma_contours: plot filled sigma contours using turbo palette
        blend_eq_lens: assume blend light == lens light
        plotname: optional plotname - pass None if you don't want to save the plot as a file
        histogram_bins: number of bins
        figsize: matplotlib figsize kwarg
        axis: pass a figure axis if you want to plot blend-lens as a subplot
        x_limits: custom limits for x axis
        y_limits: custom limits for y axis
        include_cbar: include a colorbar next to the plot

    Returns:

    """
    blend_eq_lens_msg: str = 'FOR BLEND==LENS ' if blend_eq_lens else ''

    print(f"--- PLOTTING MASS-DISTANCE PLOT {blend_eq_lens_msg}---")

    if blend_eq_lens:
        mass_lens: np.array = __filter_for_blend_eq_lens(results, "ML")
        weights: np.array = __filter_for_blend_eq_lens(results, "weights")
        dist_lens: np.array = __filter_for_blend_eq_lens(results, "DL")
    else:
        mass_lens: np.array = results.get("ML")
        weights: np.array = results.get("weights")
        dist_lens: np.array = results.get("DL")

    if axis is None:
        plt.figure(figsize=figsize)
        axis = plt.axes([.1, .15, .9, .75])

    try:
        histogram, x_edges, y_edges = np.histogram2d(mass_lens, dist_lens, weights=weights,
                                                     bins=[10 ** np.linspace(-1, 1.7, histogram_bins),
                                                           np.linspace(0, parameters["ds_median"]+parameters["ds_err_pos"],
                                                                       histogram_bins)],
                                                     density=True)

    except ValueError as e:
        print("There number of bins was too small in order to create increasing contour levels for mass-dist plot. "
              "Try another value.")
        exit(2)

    try:
        if not (peacock_sigma_contours or turbo_sigma_contours):
            __continuous_mass_dist_histogram(histogram=histogram,
                                             x_edges=x_edges,
                                             y_edges=y_edges,
                                             axis=axis,
                                             include_cbar=include_cbar)
        else:
            print(x_limits)
            if x_limits is not None:
                plt.xlim(x_limits)
            if y_limits is not None:
                plt.ylim(y_limits)
            __contour_mass_distance_histogram(histogram=histogram,
                                              x_edges=x_edges,
                                              y_edges=y_edges,
                                              color_blind_sigma_contours=turbo_sigma_contours,
                                              axis=axis,
                                              include_cbar=include_cbar)
    except ValueError as e:
        print("There number of bins was too small in order to create increasing contour levels for blend-lens plot. "
              "Try another value.")
        exit(2)

    axis.set_facecolor('black')
    axis.set_ylabel(r'lens mass $[M_\odot]$')
    axis.set_xlabel('lens distance [kpc]')
    axis.set_yscale('log')
    if x_limits is not None:
        axis.set_xlim(x_limits)
    else:
        axis.set_xlim(0, parameters["ds_median"]+parameters["ds_err_pos"])
    if y_limits is not None:
        axis.set_ylim(y_limits)
    else:
        axis.set_ylim(0.1, 50)
    axis.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if plotname:
        mass_distance_plotname = generate_plot_name(plotname, 'mass_distance')
        plt.savefig(mass_distance_plotname)
        print(f"Plot saved: {mass_distance_plotname}\n")


def blend_lens_plot(parameters: Dict[str, float],
                    results: Dict[str, float],
                    peacock_sigma_contours: bool = False,
                    turbo_sigma_contours: bool = False,
                    blend_eq_lens: bool = False,
                    plotname: Optional[str] = None,
                    histogram_bins: int = 100,
                    figsize: Tuple[int, int] = (8, 5),
                    axis: Optional[plt.axes] = None,
                    x_limits: Tuple[float, float] = None,
                    y_limits: Tuple[float, float] = None,
                    include_cbar: bool = True):
    """
    Plot the blend-lens plot
    Args:
        parameters: parameters parsed from the yaml file
        results: results loaded from samples
        peacock_sigma_contours: plot filled sigma contours using peacock palette
        turbo_sigma_contours: plot filled sigma contours using turbo palette
        blend_eq_lens: assume blend light == lens light
        plotname: optional plotname - pass None if you don't want to save the plot as a file
        histogram_bins: number of bins
        figsize: matplotlib figsize kwarg
        axis: pass a figure axis if you want to plot blend-lens as a subplot
        include_cbar: include a colorbar next to the plot

    Returns:

    """
    blend_eq_lens_msg: str = 'FOR BLEND==LENS ' if blend_eq_lens else ''

    if axis is None:
        plt.figure(figsize=figsize)
        axis = plt.axes([.1, .15, .9, .75])

    if blend_eq_lens:
        weights: np.array = __filter_for_blend_eq_lens(results, "weights")
        mag_blend: np.array = __filter_for_blend_eq_lens(results, "mag_blend")
        mag_lens_0_extinction: np.array = __filter_for_blend_eq_lens(results, "mag_lens_0_extinction")
    else:
        weights: np.array = results.get("weights")
        mag_blend: np.array = results.get("mag_blend")
        mag_lens_0_extinction: np.array = results.get("mag_lens_0_extinction")

    extinction: float = parameters.get("extinction")

    min_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.00135)
    min_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.00135)
    max_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.99865)
    max_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.99865)

    histogram, x_edges, y_edges = np.histogram2d(mag_blend, mag_lens_0_extinction, weights=weights,
                                                 bins=[np.linspace(min([min_mag_lens, min_mag_blend]),
                                                                   max(max_mag_lens,
                                                                       max_mag_blend),
                                                                   histogram_bins),
                                                       np.linspace(min([min_mag_lens, min_mag_blend]),
                                                                   max(max_mag_lens,
                                                                       max_mag_blend),
                                                                   histogram_bins)],
                                                 density=True)

    # BLEND - LENS
    print(f"--- PLOTTING BLEND-LENS PLOT {blend_eq_lens_msg}---")

    axis.set_facecolor('black')

    try:
        if not (peacock_sigma_contours or turbo_sigma_contours):
            __continuous_blend_lens_histogram(histogram=histogram,
                                              x_edges=x_edges,
                                              y_edges=y_edges,
                                              axis=axis,
                                              include_cbar=include_cbar)
        else:
            __contour_blend_lens_histogram(histogram=histogram,
                                           x_edges=x_edges,
                                           y_edges=y_edges,
                                           color_blind_sigma_contours=turbo_sigma_contours,
                                           axis=axis,
                                           include_cbar=include_cbar)

    except ValueError:
        print("There number of bins was too small in order to create increasing contour levels for blend-lens plot. "
              "Try another value.")
        exit(2)

    if x_limits is not None:
        axis.set_xlim(x_limits)
    else:
        axis.set_xlim(min([min_mag_lens, min_mag_blend]), const.MIN_MAG)

    if y_limits is not None:
        axis.set_ylim(y_limits)
    else:
        axis.set_ylim(const.MIN_MAG, min([min_mag_lens, min_mag_blend]))

    axis.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel('lens light [mag]')
    axis.set_ylabel('blend light [mag]')

    __plot_extinction_lines(extinction, axis=axis)

    if plotname:
        blend_lens_plotname = generate_plot_name(plotname, 'blend_lens')
        plt.savefig(blend_lens_plotname)
        print(f"Plot saved: {blend_lens_plotname}\n")


# --- 1D HISTOGRAMS --- #
def __plot_histogram_for_property(results: dict,
                                  property_name: str,
                                  x_linspace: np.array,
                                  x_label: str,
                                  axis: plt.axes,
                                  blend_eq_lens: bool = False):
    if blend_eq_lens:
        weights: np.array = __filter_for_blend_eq_lens(results, "weights")
        property_values: np.array = __filter_for_blend_eq_lens(results, property_name)
    else:
        weights: np.array = results.get("weights")
        property_values: np.array = results.get(property_name)

    pdf = ss.gaussian_kde(property_values, weights=weights)

    y = pdf(x_linspace)

    # Get the mode and the median
    mode = x_linspace[np.argmax(y)]
    mode_pdf = y[np.argmax(y)]
    median = quantile(property_values, weights, 0.5)
    median_pdf = pdf(median)

    # Get the 1 sigma values
    sigma1_left_bound = quantile(property_values, weights, const.SIGMA_MINUS)
    sigma1_right_bound = quantile(property_values, weights, const.SIGMA_PLUS)

    # Get the indices corresponding to the 1 sigma values
    x_sigma1_left_arg = np.argmin(np.abs(x_linspace - sigma1_left_bound))
    x_sigma1_right_arg = np.argmin(np.abs(x_linspace - sigma1_right_bound))
    axis.plot(x_linspace, y, c=HIST1D_COLOR)
    axis.set_xlabel(x_label)
    axis.set_ylabel('PDF')

    axis.vlines(median, 0, median_pdf, linewidth=1.25, color=HIST1D_COLOR)
    axis.vlines(mode, 0, mode_pdf, linewidth=1, color=HIST1D_COLOR, ls='--')

    axis.set_xlim(left=quantile(property_values, weights, 0.001), right=quantile(property_values, weights, 0.999))
    axis.set_ylim(bottom=0)

    axis.fill_between(x_linspace[x_sigma1_left_arg:x_sigma1_right_arg + 1],
                      y[x_sigma1_left_arg:x_sigma1_right_arg + 1], 0, alpha=.15, color=HIST1D_COLOR)


def plot_lens_mass_distributions(results: np.array,
                                 axis: plt.axes,
                                 blend_eq_lens: bool = False):
    max_mass_lens = quantile(results.get("ML"), results.get("weights"), 0.99865)
    __plot_histogram_for_property(results, "ML", np.linspace(0, max_mass_lens, 1000),
                                  'Lens mass [$M_\odot$]',
                                  axis, blend_eq_lens=blend_eq_lens)


def plot_lens_distance_distribution(results: np.array,
                                    axis: plt.axes,
                                    blend_eq_lens: bool = False, ):
    max_dist_lens = quantile(results.get("DL"), results.get("weights"), 0.99865)
    __plot_histogram_for_property(results, "DL", np.linspace(0, max_dist_lens, 1000),
                                  'Lens distance [kpc]',
                                  axis, blend_eq_lens=blend_eq_lens)


def plot_lens_mag_distributions(results: np.array, parameters,
                                axis: plt.axes,
                                blend_eq_lens: bool = False,
                                extinction: bool = False):
    min_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.00135)
    min_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.00135)
    max_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.99865)
    max_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.99865)

    property_name: str = "mag_lens_w_extinction" if extinction else "mag_lens_0_extinction"
    label: str = f'Lens magnitude (with max extinction {parameters.get("extinction")}) [mag]' if extinction else "Lens magnitude (no extinction) [mag]"

    __plot_histogram_for_property(results, property_name,
                                  np.linspace(min([min_mag_lens, min_mag_blend]),
                                              max(max_mag_lens, max_mag_blend), 1000),
                                  label, axis, blend_eq_lens=blend_eq_lens)


def plot_blend_mag_distributions(results: np.array,
                                 axis: plt.axes,
                                 blend_eq_lens: bool = False, ):
    min_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.00135)
    min_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.00135)
    max_mag_lens = quantile(results.get("mag_blend"), results.get("weights"), 0.99865)
    max_mag_blend = quantile(results.get("mag_lens_0_extinction"), results.get("weights"), 0.99865)

    __plot_histogram_for_property(results, "mag_blend",
                                  np.linspace(min([min_mag_lens, min_mag_blend]),
                                              max(max_mag_lens, max_mag_blend), 1000),
                                  'Blend magnitude [mag]', axis, blend_eq_lens=blend_eq_lens)

def plot_histograms(results: np.array,
                    parameters: np.array,
                    blend_eq_lens: bool = False,
                    plotname: str = None,
                    extinction: bool = False):
    blend_eq_lens_msg: str = 'FOR BLEND==LENS ' if blend_eq_lens else ''
    print(f"--- PLOTTING 1D HISTOGRAMS {blend_eq_lens_msg}---")

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    plot_lens_mass_distributions(results, ax[0][0], blend_eq_lens=blend_eq_lens)
    plot_lens_distance_distribution(results, ax[0][1], blend_eq_lens=blend_eq_lens)
    plot_lens_mag_distributions(results, parameters, ax[1][0], extinction=extinction, blend_eq_lens=blend_eq_lens)
    plot_blend_mag_distributions(results, ax[1][1], blend_eq_lens=blend_eq_lens)

    plt.tight_layout()

    mass_histogram_plotname = generate_plot_name(plotname, 'mass_histogram')
    distance_histogram_plotname = generate_plot_name(plotname, 'distance_histogram')
    histogram_plotname = generate_plot_name(plotname, 'histogram')
    plt.savefig(histogram_plotname)
    print(f"Plot saved: {histogram_plotname}\n")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_lens_mass_distributions(results, ax, blend_eq_lens=blend_eq_lens)
    plt.savefig(mass_histogram_plotname, bbox_inches='tight')
    print(f"Plot saved: {mass_histogram_plotname}\n")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_lens_distance_distribution(results, ax, blend_eq_lens=blend_eq_lens)
    plt.savefig(distance_histogram_plotname, bbox_inches='tight')
    print(f"Plot saved: {distance_histogram_plotname}\n")


# --- CORNERPLOT --- #

def cornerplot(plotname: str, res_par: Dict[str, float]):
    print(f"--- PLOTTING CORNERPLOT ---")

    corner_data = np.vstack(
        [res_par.get("ML"), res_par.get("DL"), res_par.get("mag_blend"), res_par.get("mag_lens_0_extinction"),
         res_par.get("u0")])
    figure = corner.corner(corner_data.T, labels=[r"$M_L$", r"$D_L$", r"$m_{blend}$", r"$m_{lens}$", r"$u_0$"],
                           levels=(0.6827, 0.9545, 0.9973),
                           smooth=False,
                           weights=res_par.get("weights"),
                           plot_density=False,
                           plot_datapoints=False,
                           fill_contours=True,
                           show_titles=True, title_kwargs={"fontsize": 12})
    
    corner_plotname = generate_plot_name(plotname, 'corner')
    plt.savefig(corner_plotname)
    print(f"Cornerplot saved: {corner_plotname}\n")


@click.command()
@click.argument('input-file', type=click.Path(exists=True))
@click.option('-p', '--plot-type',
              type=click.Choice(['log-density', 'sigma', 'sigma-peacock'], case_sensitive=False),
              default='log-density',
              help='Log-density is a continuous plot of log density, sigma is a contour plot. Default colorscale for '
                   'log density sigma plot turbo (color-blind friendly), '
                   'sigma plot is also available in a peacock palette. '
                   'Default type is log-density.')
@click.option('-hb', '--histogram-bins', default=100, type=click.IntRange(10, 500),
              help='Bins for the histogram plots. Default is 100.')
def main(input_file: str,
         plot_type: str,
         histogram_bins: int):
    print("--- READING INPUT PARAMETERS ---")

    with open(input_file, 'r') as input_yaml:
        settings = yaml.safe_load(input_yaml)

    parameters, samples_filename, plot_filename = parse_yaml_file(settings)

    print("--- READING SAMPLES ---")

    result_parameters: Dict[str, np.array] = read_result_parameters(samples_filename)

    # Create the "plots" directory if it doesn't exist.
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    mass_distance_plot(plotname=plot_filename,
                       parameters=parameters,
                       results=result_parameters,
                       peacock_sigma_contours=plot_type.lower() == 'sigma-peacock',
                       turbo_sigma_contours=plot_type.lower() == 'sigma',
                       blend_eq_lens=False,
                       histogram_bins=histogram_bins)

    blend_lens_plot(plotname=plot_filename,
                    parameters=parameters,
                    results=result_parameters,
                    peacock_sigma_contours=plot_type.lower() == 'sigma-peacock',
                    turbo_sigma_contours=plot_type.lower() == 'sigma',
                    blend_eq_lens=False,
                    histogram_bins=histogram_bins)

    mass_distance_plot(plotname=generate_plot_name(plot_filename, 'blend_eq_lens'),
                       parameters=parameters,
                       results=result_parameters,
                       peacock_sigma_contours=plot_type.lower() == 'sigma-peacock',
                       turbo_sigma_contours=plot_type.lower() == 'sigma',
                       blend_eq_lens=True,
                       histogram_bins=histogram_bins)

    blend_lens_plot(plotname=generate_plot_name(plot_filename, 'blend_eq_lens'),
                    parameters=parameters,
                    results=result_parameters,
                    peacock_sigma_contours=plot_type.lower() == 'sigma-peacock',
                    turbo_sigma_contours=plot_type.lower() == 'sigma',
                    blend_eq_lens=True,
                    histogram_bins=histogram_bins)

    cornerplot(plot_filename, result_parameters)

    plot_histograms(results=result_parameters,
                    parameters=parameters,
                    plotname=plot_filename)

    plot_histograms(results=result_parameters,
                    parameters=parameters,
                    plotname=generate_plot_name(plot_filename, 'blend_eq_lens'),
                    blend_eq_lens=True)


if __name__ == '__main__':
    main()
