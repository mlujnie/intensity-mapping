import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import glob
from astropy.io import ascii

from cobaya import likelihood

basedir = "/work/05865/maja_n/stampede2/master/"

psf_shape = ascii.read(basedir + "intensity-mapping/PSF/PSF.tab")
psf_shape["psf_iter"][~np.isfinite(psf_shape["psf_iter"])] = 0

# normalize such that it goes to one at r=0
psf_gaus_filt = gaussian_filter(psf_shape["psf_iter"], 2)
psf_gaus_filt /= psf_gaus_filt[0]

# normalize so that at r=0.5*FWHM, the PSF = 0.5.
psf_back_func = interp1d(psf_gaus_filt, psf_shape["r/fwhm"])
psf_shape["r/fwhm"] = psf_shape["r/fwhm"]/psf_back_func(0.5)*0.5

# filter out little bumps
psf_shape["psf_iter"][psf_shape["r/fwhm"]>3.2] = median_filter(psf_shape["psf_iter"][psf_shape["r/fwhm"]>3.2], size=20)
psf_gaus_filt = gaussian_filter(psf_shape["psf_iter"], 2)
psf_gaus_filt /= psf_gaus_filt[0]

# now interpolate
psf_func = interp1d(psf_shape["r/fwhm"], psf_gaus_filt, kind = "cubic", fill_value="extrapolate")

def fit_psf(dist, amp, fwhm):
	return psf_func(dist/fwhm) * amp

def psf_normalized(dist, fwhm):
	diff = 0.01
	x = np.arange(-6,6,diff)
	integral = np.nansum(psf_func(abs(x)/fwhm)*diff)
	func = psf_func(abs(dist)/fwhm) / integral
	return func

def powerlaw(dist, amp):
	return amp * dist**(-2.4)

def convolve(func1, func2, kwargs1, kwargs2, y):
	diff = 0.01
	x = np.arange(-6.0, 6.0, diff)
	integral = np.nansum([func1(x, **kwargs1)*func2(y0-x, **kwargs2)*diff for y0 in y], axis=1)
	return integral

def added_profile(dist, amp_pow, amp_psf, fwhm_psf): 
	return fit_psf(dist, amp_psf, fwhm_psf) + powerlaw(dist, amp_pow)

class LaeLikePlus(likelihood.Likelihood):
	def initialize(self):

		print("Analyzing all LAEs.")
		self.def_wave = np.arange(3470, 5542, 2)

		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]
		dets_laes_all = dets_laes_all[np.argsort(dets_laes_all["detectid"])]

		stardists = []
		starflux = []
		starerr = []
		i = 0
		lae_ids = []
		shot_ids = []

		for lae_id, shot_id in zip(dets_laes_all["detectid"], dets_laes_all["shotid"]):
			try:
				tab_lae = ascii.read(self.save_dir+f"lae_{lae_id}.dat")
				mask = tab_lae["r"] > 2.5
				stardists.append(tab_lae["r"].data[mask][:50])
				starflux.append(tab_lae["flux"].data[mask][:50])
				starerr.append(tab_lae["sigma"].data[mask][:50])
				lae_ids.append(lae_id)
				shot_ids.append(shot_id)

				i+=1
			except Exception as e:
				print(f"An error occurred while loading the LAE file for LAE {lae_id}:")
				print(e)
				continue

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr)
		self.lae_ids = np.array(lae_ids)
		self.shot_ids = np.array(shot_ids)

	def logp(self, **kwargs):
		amp_pow = kwargs["A_pow"] #[kwargs["A_{}".format(i)] for i in self.lae_ids] # for the power law
		amp_psf = [kwargs["Apsf_{}".format(i)] for i in self.lae_ids] # for the PSF
		fwhm_psf =  [kwargs["fwhm_{}".format(i)] for i in self.shotids] # for the PSF
		#mu_A, sigma_A = kwargs["mu_A"], kwargs["sigma_A"]

		PSF = np.array([added_profile(dist=self.stardists[i], amp_pow=amp_pow*amp_psf[i], amp_psf=amp_psf[i], 
			fwhm_psf=fwhm_psf[i]) for i in range(self.N_stars)])
		
		logp = np.nansum(- 0.5*(self.starflux - PSF)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
		# logp += np.nansum( - 0.5*(amp_pow - mu_A)**2/sigma_A**2 - 0.5*np.log(2*np.pi*sigma_A**2))
		return logp


