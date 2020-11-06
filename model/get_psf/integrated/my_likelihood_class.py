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

# now interpolate
psf_func = interp1d(psf_shape["r/fwhm"], psf_gaus_filt, kind = "cubic", fill_value="extrapolate")

def fit_psf(dist, amp, fwhm):
	return psf_func(dist/fwhm) * amp

def fit_moffat(dist, amp, fwhm):
	beta = 3.
	gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
	return amp * (1+(dist/gamma)**2)**(-1*beta)

def integrate_moffat(dist, amp, fwhm):
    """integrates the moffat function over the fiber area"""

    INTSTEP = 0.1

    dist_xy = dist/np.sqrt(2)
    gridrange = np.arange(dist_xy-0.75, dist_xy+0.75+INTSTEP, INTSTEP) # diameter of a fiber is 1.5'' -> radius = 0.75''
    xgrid = np.array([gridrange for i in range(len(gridrange))])
    ygrid = xgrid.T

    fiber_r = np.sqrt((xgrid-dist_xy)**2 + (ygrid-dist_xy)**2)
    disthere = fiber_r <= 0.75

    grid_r = np.sqrt(xgrid**2 + ygrid**2)
    grid_r[~disthere] = np.nan

    psf_grid = fit_moffat(grid_r, amp, fwhm)
    mean_psf = np.nanmean(psf_grid[disthere])
    return mean_psf

def int_moffat(dist, amp, fwhm):
    """returns integrate_moffat() for an array of distances"""
    return [integrate_moffat(x, amp, fwhm) for x in dist]

def integrate_psf(dist, amp, fwhm):
    """integrates the PSF function over the fiber area"""

    INTSTEP = 0.1

    dist_xy = dist/np.sqrt(2)
    gridrange = np.arange(dist_xy-0.75, dist_xy+0.75+INTSTEP, INTSTEP) # diameter of a fiber is 1.5'' -> radius = 0.75''
    xgrid = np.array([gridrange for i in range(len(gridrange))])
    ygrid = xgrid.T

    fiber_r = np.sqrt((xgrid-dist_xy)**2 + (ygrid-dist_xy)**2)
    disthere = fiber_r <= 0.75

    grid_r = np.sqrt(xgrid**2 + ygrid**2)
    grid_r[~disthere] = np.nan

    psf_grid = fit_psf(grid_r, amp, fwhm)
    mean_psf = np.nanmean(psf_grid[disthere])
    return mean_psf

def int_psf(dist, amp, fwhm):
    """returns integrate_psf() for an array of distances"""
    return [integrate_psf(x, amp, fwhm) for x in dist]

class StarLike(likelihood.Likelihood):
	def initialize(self):

		print("Analyzing shot with ID {}".format(self.shotid))

		with open(basedir+"radial_profiles/stars_{}/use_stars.txt".format(self.shotid), "r") as us:
			stars = [x[:-1] for x in us.readlines()]
		us.close()

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

		#PSF = np.array([int_psf(self.stardists[i], amp_par[i], fwhm_par) for i in range(self.N_stars)])
		#logp = np.nansum(- 0.5*(self.starflux - PSF)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))
	
		logp = 0
		for i in range(self.N_stars):
			PSF = int_moffat(self.stardists[i], amp_par[i], fwhm_par)
			logp += np.nansum(- 0.5*(self.starflux[i] - PSF)**2/self.starsigma[i]**2 - 0.5*np.log(2*np.pi*self.starsigma[i]**2))
		return logp

class LaeLike(likelihood.Likelihood):
	def initialize(self):

		print("Analyzing shot with ID {}".format(self.shotid))
		self.def_wave = np.arange(3470, 5542, 2)

		dets_laes_all = ascii.read(basedir+"lists/dets_laes.tab")
		dets_laes_all = dets_laes_all[dets_laes_all["vis_class"]>3]
		dets_laes_all = dets_laes_all[dets_laes_all["shotid"]==self.shotid]
		dets_laes_all = dets_laes_all[np.argsort(dets_laes_all["detectid"])]

		stardists = []
		starflux = []
		starerr = []
		i = 0
		lae_ids = []
		
		#min_len = 20
		for lae_id in dets_laes_all["detectid"]:
			try:
				tab_lae = ascii.read(self.save_dir+f"lae_{lae_id}.dat")
				order = np.argsort(tab_lae['r'].data)
				tab_lae = tab_lae[order]
				mask = tab_lae["r"] < 5.0
				stardists.append(tab_lae["r"].data[mask][:20])
				starflux.append(tab_lae["flux"].data[mask][:20])
				starerr.append(tab_lae["sigma"].data[mask][:20])
				lae_ids.append(lae_id)

		#		min_len = np.nanmin([min_len, len(tab_lae["flux"].data[mask][:20])])

				i+=1
			except Exception as e:
				print(f"An error occurred while loading the LAE file for LAE {lae_id}:")
				print(e)
				continue
	
		#for j in range(len(stardists)):
		#	stardists[j] = stardists[j][:min_len]
		#	starflux[j] = starflux[j][:min_len]
		#	starerr[j] = starerr[j][:min_len]

		self.N_stars = i
		self.starflux, self.stardists = np.array(starflux), np.array(stardists)
		self.starsigma = np.array(starerr)
		self.my_input_params = np.ravel([["A_{}".format(lae_ids[i]), "fwhm"] for i in range(self.N_stars)])
		self.lae_ids = np.array(lae_ids)

	def logp(self, **kwargs):
		amp_par = [kwargs["A_{}".format(i)] for i in self.lae_ids] 
		fwhm_par =  kwargs["fwhm"]

		#PSF = np.array([int_psf(self.stardists[i], amp_par[i], fwhm_par) for i in range(self.N_stars)])
		#logp = np.nansum(- 0.5*(self.starflux - PSF)**2/self.starsigma**2 - 0.5*np.log(2*np.pi*self.starsigma**2))

		logp = 0
		for i in range(self.N_stars):
			PSF = int_psf(self.stardists[i], amp_par[i], fwhm_par)
			logp += np.nansum(- 0.5*(self.starflux[i] - PSF)**2/self.starsigma[i]**2 - 0.5*np.log(2*np.pi*self.starsigma[i]**2))
		return logp


