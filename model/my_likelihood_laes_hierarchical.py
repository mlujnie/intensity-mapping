import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import glob
from astropy.io import ascii

#from hetdex_api.shot import *

from cobaya import likelihood
#from cobaya.likelihoods._base_classes import _DataSetLikelihood

def load_shot(shot):
		fileh = open_shot_file(shot, survey="hdr2.1")
		table = Table(fileh.root.Data.Fibers.read())
		fileh.close()
		return table

basedir = "/work/05865/maja_n/stampede2/master/"

def fit_powerlaw(r, amp):
	return amp * r**(-2.4)

def distsq(midra, middec, ras, decs):
	"""returns the angular distances in arcseconds"""
	deccos = np.cos(middec*np.pi/180.)
	return (((ras-midra)*deccos)**2 + (decs-middec)**2)*(3600)**2

class HierLike(likelihood.Likelihood):
	def initialize(self):

		self.def_wave = np.arange(3470, 5542, 2)

		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]
		dets_laes_all = dets_laes_all[np.argsort(dets_laes_all["detectid"])]

		starmids = []
		stardists = []
		starflux = []
		starerr = []
		i = 0
		lae_ids = []

		for lae_id in dets_laes_all["detectid"]:
			try:
				tab_lae = ascii.read(self.save_dir+f"lae_{lae_id}.dat")
				mask = tab_lae["r"] > 2.5
				stardists.append(tab_lae["r"].data[mask])
				starflux.append(tab_lae["flux"].data[mask])
				starerr.append(tab_lae["sigma"].data[mask])
				lae_ids.append(lae_id)

				i+=1
			except Exception as e:
				print(f"An error occurred while loading the LAE file for LAE {lae_id}:")
				print(e)
				continue

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr)
		self.my_input_params = ["A_{}".format(i) for i in range(self.N_stars)]
		self.lae_ids = np.array(lae_ids)

	def logp(self, **kwargs):
		amp_par = np.array([kwargs["A_{}".format(i)] for i in self.lae_ids])
		mu_A, sigma_A = kwargs["mu_A"], kwargs["sigma_A"]

		profile = np.array([fit_powerlaw(self.stardists[i], amp_par[i]) for i in range(self.N_stars)])

		logp = np.nansum(- 0.5*(self.starflux - profile)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
		logp += np.nansum( - 0.5*(amp_par - mu_A)**2/sigma_A**2 - 0.5*np.log(2*np.pi*sigma_A**2))
		return logp
