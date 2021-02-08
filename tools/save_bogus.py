#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import random

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
	
	randnum = random.randint(0, len(shot_tab))
	d_ra = random.uniform(-1, 1)/3600.
	d_dec = random.uniform(-1, 1)/3600.
	d_wave = random.uniform(-50, 50)

	lae_ra, lae_dec = shot_tab["ra"][randnum]+d_ra, shot_tab["dec"][randnum]+d_dec

	lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
	rs = lae_coords.separation(shot_coords)
	mask = rs <= 10*u.arcsec
	rs = rs[mask]
	spec_here = ffskysub[mask]
	err_here = spec_err[mask]
	mask_7_here = mask_7[mask]
	mask_10_here = mask_10[mask]
	order = np.argsort(rs)
	rs = rs[order]
	spec_here = spec_here[order]
	err_here = err_here[order]
	mask_7_here = mask_7_here[order]
	mask_10_here = mask_10_here[order]

	#continuum = spec_here.copy()
	#continuum = continuum.T
	#for i in range(1036):
	#	continuum[i] = np.nanmedian(continuum[filter_min[i]:filter_max[i]+1], axis=0)
	#continuum = continuum.T
	#continuum_subtracted = spec_here.copy() - continuum
	
	lae_wave = lae["wave"] + d_wave
	lae_linewidth = lae["linewidth"]
	wlhere = abs(def_wave - lae_wave) <= 1.5 * lae_linewidth
		
	spec_sum = np.nansum(spec_here[:,wlhere], axis=1)
	err_sum = np.sqrt(np.nansum(err_here[:,wlhere]**2, axis=1))

	#spec_sub_sum = np.nansum(continuum_subtracted[:,wlhere], axis=1)
	
	mask = (spec_sum != 0) & (err_sum != 0)
	rs = rs[mask][:] / u.arcsec
	rs = rs.decompose()

	spec_sum = spec_sum[mask].data[:]
	err_sum = err_sum[mask].data[:]
	#spec_sub_sum = spec_sub_sum[mask].data[:]
	mask_7_here = mask_7_here[mask]
	mask_10_here = mask_10_here[mask]
	
	tab = {"r": rs,
		"flux":spec_sum,
	#	"flux_contsub":spec_sub_sum,
		"sigma": err_sum,
		"mask_7": mask_7_here,
		"mask_10": mask_10_here}
		
	save_file = os.path.join(basedir, f"radial_profiles/bogus_laes/bogus_{detectid}.dat")
	ascii.write(tab, save_file)
	print("Wrote to "+save_file)
	return 1

   
basedir = "/work/05865/maja_n/stampede2/master"
complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "high_sn_sources.tab"))
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

def_wave = np.arange(3470, 5542, 2.)
a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0)                                 
b = np.min([np.ones(1036)*1035, a+190], axis=0)                                             
c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0)
filter_min = np.array(c, dtype=int)
filter_max = np.array(b, dtype=int)
for shotid in np.unique(complete_lae_tab["shotid"])[::-1]:
	#load the shot table and prepare full-frame sky subtracted spectra
	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	done = True 
	for detectid in laes_here["detectid"].data:
		done *= os.path.exists(os.path.join(basedir, f"radial_profiles/bogus_laes/bogus_{detectid}.dat"))
	if done:
		print("Already finished", shotid)
		continue
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
	#ffskysub[abs(medians_lo)>perc_lo] *= np.nan
	#ffskysub[abs(medians_hi)>perc_hi] *= np.nan
	mask_7 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)

	# exclude extreme continuum values
	perc = 90
	
	wlcont_lo = (def_wave > 4000)&(def_wave <= 4500)
	medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1)
	perc_lo = np.nanpercentile(medians_lo, perc)

	wlcont_hi = (def_wave > 4800)&(def_wave <= 5300)
	medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1)
	perc_hi = np.nanpercentile(medians_hi, perc)
	#ffskysub[abs(medians_lo)>perc_lo] *= np.nan
	#ffskysub[abs(medians_hi)>perc_hi] *= np.nan
	mask_10 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)


	spec_err = shot_tab["calfibe"].copy()
	spec_err[~np.isfinite(ffskysub)] = np.nan

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)
	for detectid in laes_here["detectid"].data:
		if os.path.exists(os.path.join(basedir, f"radial_profiles/bogus_laes/bogus_{detectid}.dat")):
			continue
		tmp = save_lae(detectid)
	print(f"Finished {shotid}.")	


