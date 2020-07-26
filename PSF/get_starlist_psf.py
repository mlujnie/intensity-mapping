# contains the code to choose which stars to use for getting the PSF

import tables as tb
import numpy as np

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

from hetdex_api.survey import Survey, FiberIndex

survey = Survey('hdr2')
survey_table=survey.return_astropy_table()

# open erin's list of stars in hetdex
erin_stars = ascii.read("/work/05350/ecooper/stampede2/hdr2-tests/sdss_star_hdr2.cat")
usethese = []
# only choose stars brighter than magnitude 21
for sid in sids:
    if erin_stars[erin_stars["ID"]==sid]["g"].data <= 21:
        usethese.append(sid)

# get the shots, exclude 2017
allshots = []
i = 0
really_keep_these = {}
for sid in usethese:
    this = erin_stars[erin_stars["ID"]==sid]
    coords = SkyCoord(this["ra"]*u.deg, this["dec"]*u.deg, frame='icrs')
    shots = np.array(survey.get_shotlist(coords, width=0.5, height=0.2))
    if len(shots[shots>20180000000]) > 0:
        i+=1
        really_keep_these[sid] = shots[shots>20180000000]
        allshots.append(shots[shots>20180000000])
allshots = np.concatenate(allshots)

keep_these_ids, ras, decs = [], [], []
shots = []
for sid in really_keep_these.keys():
    this = erin_stars[erin_stars["ID"]==sid]
    for shot in really_keep_these[sid]:
        keep_these_ids.append(sid)
        ras.append(this["ra"].data)
        decs.append(this["dec"].data)
        shots.append(shot)

ascii.write({
    "ID": keep_these_ids,
    "ra": ras,
    "dec": decs,
    "shotid": shots}, "analysis_stars.dat")
