from astropy.io import ascii
import numpy as np
from scipy.interpolate import interp1d

basedir = "/work/05865/maja_n/stampede2/master/"

def fwhm_prior(**kwargs):
	prior = 1.
	for key in kwargs.keys():
		if not key[:4] == "fwhm":
			continue
		shotid = key[5:]
		fwhm = kwargs[key]
		fwhm_tab = ascii.read(basedir + f"fwhm_posteriors/fwhm_{shotid}.dat")
		prior *= interp1d(fwhm_tab["fwhm"], fwhm_tab["posterior"], fill_value="extrapolate")(fwhm)

	return np.log(prior)

def amp_prior(**kwargs):
	prior = 1.
	for key in kwargs.keys():
		if not key[:4] == "Apsf":
			continue
		detectid = key[5:]
		amp = kwargs[key]
		amp_tab = ascii.read(basedir + f"lae_psf_posterior/Apsf_{detectid}.dat")
		prior *= interp1d(amp_tab["A"], fwhm_tab["posterior"], fill_value="extrapolate")(amp)

	return np.log(prior)


