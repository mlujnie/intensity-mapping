import sys
import os
import os.path
import subprocess
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u

from hetdex_api.config import HDRconfig
from hetdex_api.detections import Detections
from hetdex_api.elixer_widget_cls import ElixerWidget

# open detection catalogue
detects = Detections(survey='hdr2', catalog_type='lines')

# open LAE list 
laes = ascii.read("/work/05865/maja_n/wrangler/im2d/all_good_LAEs_060919.tab")
laes = laes[laes["cat2"] == "good"]

for i in range(len(laes)):
    ra, dec, wave = laes[i]["ra"], laes[i]["dec"], laes[i]["wl"]
    
    # set object coordinates and wavelength
    obj_coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    wave_obj = wave

    # find match in the catalogue
    idx = detects.find_match(obj_coords, wave=wave_obj, radius=5.*u.arcsec, dwave=5 )
    if len(idx[idx])==0:
        print(str(laes[i]["detectid"])+" 0000000000")
    for x in detects.detectid[idx]:
        print(str(laes[i]["detectid"])+" "+str(x))