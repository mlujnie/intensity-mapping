import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import sys
from weighted_biweight import biweight_location_weights_karl

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
    
erin_stars = ascii.read("/data/05865/maja_n/Jupyter/analysis_stars.dat")
order = np.argsort(erin_stars["shotid"])[::-1]
erin_stars = erin_stars[order]

def_wave = np.arange(3470, 5542, 2.)
wlhere = np.where((def_wave>=4450)&(def_wave<=4550))[0]

def save_stars_in_shot(shot_idx):
    shot = shotlist[shot_idx] # erin_stars["shotid"][idx]  
    stars_shot = erin_stars[erin_stars["shotid"] == shot]
    table = load_shot(shot)
    table["spec_fullsky_sub"][table["spec_fullsky_sub"]==0] = np.nan
    table["calfibe"][table["spec_fullsky_sub"]==0] = np.nan
    weights = np.ones(table["spec_fullsky_sub"].shape)
    weights[~np.isfinite(table["spec_fullsky_sub"])] = np.nan
    #print("opened shot ", shot)
    for star_idx in range(len(stars_shot)):
        star_ID = stars_shot["ID"][star_idx]
        ff = glob.glob("/data/05865/maja_n/Jupyter/use_stars/star_{}_{}.tab".format(shot, star_ID))
        #if len(ff) > 0:
        #    continue
        try:
            coords = SkyCoord(stars_shot["ra"][star_idx]*u.deg, stars_shot["dec"][star_idx]*u.deg)
            ras, decs = table["ra"], table["dec"]
            n_here = ((ras-coords.ra.value)*np.cos(coords.dec.value*np.pi/180))**2+(decs-coords.dec.value)**2 < (13./3600)**2
            if len(n_here[n_here])<5:
                continue
            else:
                #mids = np.nanmean(table["calfib"][n_here][:,wlhere], axis=1)
                #stds = np.nanstd(table["calfib"][n_here][:,wlhere], axis=1)/np.sqrt(len(wlhere))
                mids = biweight_location_weights_karl(table["spec_fullsky_sub"][n_here][:,wlhere], weights = weights[n_here][:,wlhere] , axis=1)
            
                stds = biweight_scale(table["spec_fullsky_sub"][n_here][:,wlhere], axis=1, ignore_nan=True)/np.sqrt((len(wlhere)-1))
                tab = Table({"ra": ras[n_here], "dec": decs[n_here], "flux":mids, "std":stds, "star_ra": [coords.ra.value for j in range(len(mids))], "star_dec": [coords.dec.value for j in range(len(mids))]})
                ascii.write(tab, "/data/05865/maja_n/Jupyter/ff2.1_stars/star_{}_{}.tab".format(shot, star_ID), comment=True)
                #print("Wrote to new_startabs/star_{}_{}.tab".format(shot, star_ID))
        except:
            pass
    return 1

shotlist = np.unique(erin_stars["shotid"])
n = len(shotlist)

cpu_count = multiprocessing.cpu_count()
print("Number of available CPUs: ", cpu_count)
pool = multiprocessing.Pool(6)

print("Starting to run.")

start = time.time()
endlist = pool.map(save_stars_in_shot, np.arange(0, n))
end = time.time()
print("Time needed: "+str(end-start))
