from astropy.io import ascii
from scipy.interpolate import interp1d

fwhm_tab = ascii.read("fwhm.dat")

def fwhm_prior(fwhm):
	return np.log(interp1d(fwhm_tab["fwhm"], fwhm_tab["posterior"], fill_value="extrapolate")(fwhm))
