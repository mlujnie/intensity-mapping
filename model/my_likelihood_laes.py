import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import glob
from astropy.io import ascii

from hetdex_api.shot import *

from cobaya import likelihood

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

class MyLike(likelihood.Likelihood):
	def initialize(self):

		self.shotids = [20200107026,
				20200124019,
				20200124020,
				20200228022,
				20200228026,
				20200228027,
				20200228028,
				20200228029,
				20200425010,
				20200425013,
				20200430025,
				20200430026,
				20200514021,
				20200626016]

		self.def_wave = np.arange(3470, 5542, 2)
		
		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]

		starmids = []
		stardists = []
		starflux = []
		starerr = []
		i = 0

		for shotid in self.shotids:
			dets_laes = dets_laes_all[dets_laes_all["shotid"]==shotid]
			if len(dets_laes)==0:
				continue

			# load the shot table and prepare full-frame sky subtracted spectra
			try:
				shot_tab = load_shot(shotid)
			except Exception as e:
				print(f"Could not load shot {shotid}. Error message:")
				print(e)
				continue
			ffskysub = shot_tab["spec_fullsky_sub"].copy()
			ffskysub[ffskysub==0] = np.nan

			# exclude extreme continuum values
			wlcont = (self.def_wave > 4000)&(self.def_wave <= 4500)
			medians = np.nanmedian(ffskysub[:,wlcont], axis=1)
			perc = np.percentile(medians, 95)
			ffskysub[abs(medians)>perc] *= np.nan

			spec_err = shot_tab["calfibe"].copy()
			spec_err[~np.isfinite(ffskysub)] = np.nan
			for lae in dets_laes:
				lae_ra, lae_dec = lae["ra"], lae["dec"]
				rs = np.sqrt(distsq(lae_ra, lae_dec, shot_tab["ra"], shot_tab["dec"]))

				lae_wave = lae["wave"]
				wlhere = abs(self.def_wave - lae_wave) < 3.

				mask = (rs >= 2.5) & (rs <= 10)
				rs = rs[mask]
				spec_here = ffskysub[mask]
				err_here = spec_err[mask]

				order = np.argsort(rs)
				rs = rs[order]
				spec_here = spec_here[order]
				err_here = err_here[order]

				spec_here = np.nansum(spec_here[:,wlhere], axis=1)			
				err_sum = np.sqrt(np.nansum(err_here[:,wlhere]**2, axis=1))

				mask = (spec_here != 0) & (err_sum != 0)
				rs = rs[mask].data[:50]
				spec_here = spec_here[mask][:50]
				err_sum = err_sum[mask][:50]

				stardists.append(rs)
				starflux.append(spec_here)
				starerr.append(err_sum)

				i+=1

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr) 
		self.my_input_params = ["A_{}".format(i) for i in range(self.N_stars)]

	def logp(self, **kwargs):
		amp_par = [kwargs["A_{}".format(i)] for i in range(self.N_stars)] 

		profile = np.array([fit_powerlaw(self.stardists[i], amp_par[i]) for i in range(self.N_stars)])
		
		logp = np.nansum(- 0.5*(self.starflux - profile)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
		return logp

