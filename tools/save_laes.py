#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import sys
import os
from weighted_biweight import biweight_location_weights_karl

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
 
def save_lae(detectid):
	lae = complete_lae_tab[complete_lae_tab["detectid"]==detectid]
	
	lae_ra, lae_dec = lae["ra"], lae["dec"]
	lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
	rs = lae_coords.separation(shot_coords)
	mask = rs <= 10*u.arcsec
	rs = rs[mask]
	spec_here = ffskysub[mask]
	err_here = spec_err[mask]
	order = np.argsort(rs)
	rs = rs[order]
	spec_here = spec_here[order]
	err_here = err_here[order]

	lae_wave = lae["wave"]
	lae_linewidth = lae["linewidth"]
	wlhere = abs(def_wave - lae_wave) <= 1.5 * lae_linewidth
		
	spec_sum = np.nansum(spec_here[:,wlhere], axis=1)
	err_sum = np.sqrt(np.nansum(err_here[:,wlhere]**2, axis=1))
	
	mask = (spec_sum != 0) & (err_sum != 0)
	rs = rs[mask][:100] / u.arcsec
	rs = rs.decompose()
	print("rs: ", rs)
	spec_sum = spec_sum[mask].data[:100]
	err_sum = err_sum[mask].data[:100]
	
	tab = {"r": rs,
		"flux":spec_sum,
		"sigma": err_sum}
	save_file = os.path.join(basedir, f"radial_profiles/laes/lae_{detectid}.dat")
	ascii.write(tab, save_file)
	print("Wrote to "+save_file)
	return 1

   
basedir = "/work/05865/maja_n/stampede2/master"
complete_lae_tab = ascii.read(os.path.join(basedir, "good_LAEs_classified.tab"))
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

def_wave = np.arange(3470, 5542, 2.)

for shotid in np.unique(complete_lae_tab["shotid"]):
	#load the shot table and prepare full-frame sky subtracted spectra
	try:
		shot_tab = load_shot(shotid)
	except Exception as e:
		print(f"Could not load shot {shotid}. Error message:")
		print(e)
		continue
	ffskysub = shot_tab["spec_fullsky_sub"].copy()
	ffskysub[ffskysub==0] = np.nan

	# exclude extreme continuum values
	perc = 93
	
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

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)

	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	for detectid in laes_here["detectid"].data:
		tmp = save_lae(detectid)
	print(f"Finished {shotid}.")	


