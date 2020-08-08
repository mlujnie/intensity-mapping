"""This code is step 1 of the Lyman-alpha halo analysis.
We collect and save the radial profiles of stars in the specified shot."""

from astropy.io import ascii
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import glob
from astropy.stats import biweight_midvariance, biweight_location
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shotid", type=int, default=None,
                    help="Shotid.")
args = parser.parse_args(sys.argv[1:])

if args.shotid is None:
    sys.exit("Please enter a shotid with\npython3 get_star_data.py -s $SHOTID")
shotid  = args.shotid

basedir = "/work/05865/maja_n/stampede2/master/"

from hetdex_api.shot import *
def load_shot(shot):
        fileh = open_shot_file(shot, survey="hdr2.1")
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table

def distsq(midra, middec, ras, decs):

    """returns the angular distances in arcseconds"""

    deccos = np.cos(middec*np.pi/180.)
    return (((ras-midra)*deccos)**2 + (decs-middec)**2)*(3600)**2

shot_tab = load_shot(shotid)
ffskysub = shot_tab["spec_fullsky_sub"]
ffskysub[ffskysub==0] = np.nan
def_wave = np.arange(3470,5542, 2)

karl_stars = ascii.read(basedir + "lists/KarlsCalStars.tab")
stars = karl_stars[karl_stars["shotid"] == shotid]

psf_shape = ascii.read(basedir+"intensity-mapping/PSF/PSF.tab")
#psf_shape = psf_shape[np.isfinite(psf_shape["psf_iter"])]
psf_shape["psf_iter"][~np.isfinite(psf_shape["psf_iter"])] = 0

def fit_psf(dist, amp, fwhm):
    return psf_func(dist/fwhm) * amp

# normalize such that it goes to one at r=0
psf_gaus_filt = gaussian_filter(psf_shape["psf_iter"], 2)
psf_gaus_filt /= psf_gaus_filt[0]

# normalize so that at r=0.5*FWHM, the PSF = 0.5.
psf_back_func = interp1d(psf_gaus_filt, psf_shape["r/fwhm"])
psf_shape["r/fwhm"] = psf_shape["r/fwhm"]/psf_back_func(0.5)*0.5

# now interpolate
psf_func = interp1d(psf_shape["r/fwhm"], psf_gaus_filt, kind = "cubic", fill_value="extrapolate")

wave_here = (def_wave > 4550)&(def_wave <= 4650)
for star in stars:
    detectid = star["detectid"]
    ra, dec = star["ra_1"], star["dec_1"]
    rsqs = distsq(ra, dec, shot_tab["ra"], shot_tab["dec"])
    mask_here = rsqs < 10**2

    these_fibers = biweight_location(ffskysub[mask_here][:,wave_here], ignore_nan=True, axis=1)
    these_errs = np.sqrt(biweight_midvariance(ffskysub[mask_here][:,wave_here], ignore_nan=True, axis=1))
    rs = np.sqrt(rsqs[mask_here])

    p0 = [np.nanmax(these_fibers), 1.3]
    mask_finite = np.isfinite(these_fibers) & np.isfinite(these_errs)
    popt, pcov = curve_fit(fit_psf, rs[mask_finite], these_fibers[mask_finite], sigma=these_errs[mask_finite], p0=p0)
    if np.sqrt(pcov[1,1]) > 0.1:
        continue

    star_dict = {"r": rs, "flux": these_fibers, "sigma": these_errs}
    ascii.write(star_dict, basedir + f"radial_profiles/stars_{shotid}/{detectid}.dat")
