import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import glob
from astropy.io import ascii

from hetdex_api.shot import *

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

class MyLike(likelihood.Likelihood):
	def initialize(self):

		self.def_wave = np.arange(3470, 5542, 2)
		
		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]
		
		print("analyzing LAE ID: ", self.lae_id) 
		lae = dets_laes_all[dets_laes_all["detectid"]==self.lae_id][0]
		shotid = lae["shotid"]

		starmids = []
		stardists = []
		starflux = []
		starerr = []
		i = 0

		if self.load_saved:
			try:
				tab_lae = ascii.read(self.save_dir+f"lae_{self.lae_id}.dat")
				mask = tab_lae["r"] > 2.5
				stardists.append(tab_lae["r"].data[mask])
				starflux.append(tab_lae["flux"].data[mask])
				starerr.append(tab_lae["sigma"].data[mask])

				i+=1
				self.N_stars = i
				self.starflux, self.stardists = np.array(starflux), np.array(stardists)
				self.starsigma = np.array(starerr) 
				self.my_input_params = ["A_{}".format(i) for i in range(self.N_stars)]


			except Exception as e:
				print("An error occurred while loading the LAE file:")
				print(e)
				print("getting the radial profile from the shot instead")
				self.load_from_shot()
		else:
			self.load_from_shot()


	def load_from_shot(self):

		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]
		
		print("analyzing LAE ID: ", self.lae_id) 
		lae = dets_laes_all[dets_laes_all["detectid"]==self.lae_id][0]
		shotid = lae["shotid"]


		starmids = []
		stardists = []
		starflux = []
		starerr = []
		i = 0
	
		# load the shot table and prepare full-frame sky subtracted spectra
		try:
			shot_tab = load_shot(shotid)
		except Exception as e:
			print(f"Could not load shot {shotid}. Error message:")
			sys.exit("Exiting.")
		ffskysub = shot_tab["spec_fullsky_sub"].copy()
		ffskysub[ffskysub==0] = np.nan

		# exclude extreme continuum values
		wlcont_lo = (def_wave > 4000)&(def_wave <= 4500)
		medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1)
		perc_lo = np.nanpercentile(medians_lo, perc)

		wlcont_hi = (def_wave > 4800)&(def_wave <= 5300)
		medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1)
		perc_hi = np.nanpercentile(medians_hi, perc)
		ffskysub[abs(medians_lo)>perc_lo] *= np.nan
		ffskysub[abs(medians_hi)>perc_hi] *= np.nan

		spec_err = shot_tab["calfibe"].copy()
		spec_err[~np.isfinite(ffskysub)] = np.nan
		
		lae_ra, lae_dec = lae["ra"], lae["dec"]
		rs = np.sqrt(distsq(lae_ra, lae_dec, shot_tab["ra"], shot_tab["dec"]))

		lae_wave = lae["wave"]
		wlhere = abs(self.def_wave - lae_wave) < 3.

		mask =  (rs <= 10) # & (rs >= 2.5) 
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
		rs = rs[mask].data[:100]
		spec_here = spec_here[mask][:100]
		err_sum = err_sum[mask][:100]

		stardists.append(rs)
		starflux.append(spec_here)
		starerr.append(err_sum)

		i+=1

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr) 
		self.my_input_params = ["A_{}".format(i) for i in range(self.N_stars)]

		# save it
		self.save_radial_profile()

	def save_radial_profile(self):
		tab = {"r": self.stardists[0], "flux": self.starflux[0], "sigma": self.starsigma[0]}
		ascii.write(tab, self.save_dir + f"lae_{self.lae_id}.dat", overwrite=True)
		print(f"Wrote to {self.save_dir}lae_{self.lae_id}.dat")

	def logp(self, **kwargs):
		amp_par = [kwargs["A_{}".format(i)] for i in range(self.N_stars)] 

		profile = np.array([fit_powerlaw(self.stardists[i], amp_par[i]) for i in range(self.N_stars)])
		
		logp = np.nansum(- 0.5*(self.starflux - profile)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
		return logp

