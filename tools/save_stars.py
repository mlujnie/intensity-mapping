from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord

from hetdex_api.shot import *

from astropy.io import ascii
import glob

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close
        return table
    
erin_stars = ascii.read("stars_faint.tab")
order = np.argsort(erin_stars["shotid"])[::-1]
erin_stars = erin_stars[order]

def_wave = np.arange(3470, 5542, 2.)
wlhere = np.where((def_wave>=4450)&(def_wave<=4550))[0]

old_shot = 1000000000
for idx in range(2812,len(erin_stars),1):
    shot = erin_stars["shotid"][idx]   
    star_ID = erin_stars["Erin_id"][idx]
    if len(glob.glob("new_startabs/star_{}_{}.tab".format(shot, star_ID)))>0:
        print("yay.")
        continue
    
    if shot != old_shot:
        table = load_shot(shot)
    print("opened shot ", shot)
    old_shot = shot
    
    coords = SkyCoord(erin_stars["ra"][idx]*u.deg, erin_stars["dec"][idx]*u.deg)
    ras, decs = table["ra"], table["dec"]
    n_here = ((ras-coords.ra.value)*np.cos(coords.dec.value*np.pi/180))**2+(decs-coords.dec.value)**2 < (13./3600)**2
    if len(n_here[n_here])<5:
        print(idx)
        continue
    else:
        mids = np.nanmean(table["calfib"][n_here][:,wlhere], axis=1)
        stds = np.nanstd(table["calfib"][n_here][:,wlhere], axis=1)/np.sqrt(len(wlhere))
        tab = Table({"ra": ras[n_here], "dec": decs[n_here], "flux":mids, "std":stds, "star_ra": [coords.ra.value for j in range(len(mids))], "star_dec": [coords.dec.value for j in range(len(mids))]})
        ascii.write(tab, "new_startabs/star_{}_{}.tab".format(shot, star_ID), comment=True)
