# DarkLensCode
Code to generate mass-distance plots for microlensing events.

## Installation with pip
Install darklens with pip. For now this can be done only locally:
``` sh
python3 -m pip install -e .
```

## Manual installation

Dependencies can be installed with pip.
``` sh
python3 -m pip install -r requirements.txt
```
It is recommended to use a virtual environment.
## How to use

### Running code

To use the Dark Lens Code you need a file with chains from
the lensing model in .npy format and a .txt file with the parameters of the event. 

The results will be saved to files with results_filename and prefixes.

Having them, you can run the code.
``` sh
darklens chain_file.npy parameters_file.txt results_filename
```

### File with chains

File with chains in .npy format should contain a tuple of tuples of lens parameters.
The parameters should be in the following order
```
t0 [JD − 2450000]   u0  tE [days]   piEN    piEE    mag0    fs
```

### File with parameters

File with parameters in .txt format should contain event parameters in the following order
```
alpha [deg]   delta [deg]   t0par [JD − 2450000]   extintion   dist_s [kpc]  dist_s_max [kpc]  dist_s_min [kpc]  mu_ra [mas/yr]   mu_ra_sig [mas/yr]   mu_dec [mas/yr]  mu_dec_sig [mas/yr]  mu_ra_dec_corr
```
Parameters should be separated with tabulator.
If you know distance to the source dist_s_max = dist_s + sigma and dist_s_min = dist_s - sigma.
Otherwise, look at --ds_weight flag.

### Flags

You can use flags to run code with specific options.
Flags should be placed between calling the program and giving its arguments.
Example:
``` sh
darklens --niter 1e7 chain_file.npy parameters_file.txt plotname
```
#### ds_weight
Use when you do not know the distance to the source and 
want to weigh it with a galaxy model.
In this case dist_s_min and dist_s_max are the ends of the distance sampling range.
The default is False.
``` sh
--ds_weight True
```
#### niter
Use to change number of iterations. The default 1e6.
``` sh
--niter 5e6
```
#### masspower
Use to change mass function slope power index. The default -1.75. To be more precise, this means that mass function g(M)=M^masspower.
``` sh
--masspower -2.00
```
#### filter
Use to choose filter used in your model. Filters available: V, I, G, K.
The default G.
``` sh
--filter V
```
#### samples
Use to save samples in .npy format
``` sh
--samples True
```
Samples colums:
```
weight   M_L [Solar masses]   D_L [kpc]  mag_blend   mag_lens_with_extinction  mag_lens_without_extinction  mag_source  t0 [JD - 2450000]   tE [days]   u0  piEN  piEE  mag0	fs	mu_rel [mas/yr]
```

## Plotting

After running the ```darklens``` command, a csv file with results parameters will be saved.
These parameters can be used to plot the results using the command:

``` sh
darklens-plot [options] yaml_filename
```

The YAML file has to contain:

- **samples filename**- result of the ``darklens`` command
- **plot filename**- prefix and extension of the plot filenames
- **parameters**

The plots will be saved in a **plots** directory, created if non-existent. 

### Choosing the color

A continuous mode is available, for which the colors correspond to log prob density.
The turbo colormap is color-blind friendly.

![log_prob_colormap](readme_img/log_prob.png)

and two sigma contour palettes, which correspond to sigma values- peacock colormap and turbo (color-blind friendly) colormap.

![peacock_sigma_colormap](readme_img/peacock_sigmas.png)
![turbo_sigma_colormap](readme_img/turbo_sigmas.png)

### Options

#### plot type
Plot the distributions either using the continuous log density scale or 
in a discrete colormap according to the sigma values.

Available colormaps for the sigma values are turbo (``sigma`` option) and peacock (``sigma-peacock`` option).
Log density (``log-density`` option) plot uses the turbo colormap. It is the default option.
``` sh
-p [log-density|sigma|sigma-peacock]
--plot-type [log-density|sigma|sigma-peacock]
```
#### bins
Binning can be changed for mass-distance and blend-lens plots. The defaults are all 100.

Too small binning values can lead to an error -- the contour levels must be increasing which might be impossible for
a small number of bins.

Accepted range is [10, 500], for other values the default value of 100 will be used.

``` sh
-hb 50
--histogram-bins 50
```
