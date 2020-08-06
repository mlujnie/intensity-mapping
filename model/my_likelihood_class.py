import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import glob
from astropy.io import ascii

from cobaya import likelihood

psf_shape = ascii.read("/data/05865/maja_n/intensity-mapping/PSF/PSF.tab")
psf_shape["psf_iter"][~np.isfinite(psf_shape["psf_iter"])] = 0

		# normalize such that it goes to one at r=0
psf_gaus_filt = gaussian_filter(psf_shape["psf_iter"], 2)
psf_gaus_filt /= psf_gaus_filt[0]

# normalize so that at r=0.5*FWHM, the PSF = 0.5.
psf_back_func = interp1d(psf_gaus_filt, psf_shape["r/fwhm"])
psf_shape["r/fwhm"] = psf_shape["r/fwhm"]/psf_back_func(0.5)*0.5

# now interpolate
psf_func = interp1d(psf_shape["r/fwhm"], psf_gaus_filt, kind = "cubic", fill_value="extrapolate")

def fit_psf(dist, amp, fwhm):
	return psf_func(dist/fwhm) * amp

class MyLike(likelihood.Likelihood):
	def initialize(self):

		self.shotid = 20200124020

		stars = glob.glob("/data/05865/maja_n/radial_profiles/stars_{}/*".format(self.shotid))
		stars = np.sort(stars)

		starmids = []
		stardists = []
		starflux = []
		starerr = []
		i = 0
		for sf in stars:
			a = ascii.read(sf)
			rs = a["r"]
			order = np.argsort(rs)[:20]
			rs = rs[order]
			a = a[order]
			if np.nanmax(a["flux"]) < 2.0:
				continue

			stardists.append(rs)
			starflux.append(a["flux"].data)
			starerr.append(a["sigma"].data)

			i+=1

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr) 
		self.my_input_params = np.ravel([["A_{}".format(i), "fwhm"] for i in range(self.N_stars)])

	def logp(self, **kwargs):
		amp_par = [kwargs["A_{}".format(i)] for i in range(self.N_stars)] 
		fwhm_par =  kwargs["fwhm"]

		PSF = np.array([fit_psf(self.stardists[i], amp_par[i], fwhm_par) for i in range(self.N_stars)])
		
		logp = np.nansum(- 0.5*(self.starflux - PSF)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
		return logp

