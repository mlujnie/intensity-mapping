import astropy.units as u
from astropy.coordinates import SkyCoord

from hetdex_api.shot import *
from hetdex_tools import galmask

def load_shot(shot): 
	fileh = open_shot_file(shot, survey="hdr2.1") 
	table = Table(fileh.root.Data.Fibers.read()) 
	fileh.close() 
	return table

shot_tab = load_shot(20200423019)

tboth = galmask.read_rc3_tables()
tdetect = galmask.read_detect()

flag = []
for i in range(len(shot_tab)): 
	tmp = shot_tab[i] 
	coords = SkyCoord(ra=tmp["ra"]*u.deg, dec=tmp["dec"]*u.deg) 
	isclose, name, zgal = galmask.gals_flag_from_coords(coords, tboth, d25scale=1.75, nmatches=3) 
	flag.append(isclose) 
	print(isclose)
