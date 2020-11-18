import glob
from astropy.io import ascii
import numpy as np
import sys 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shotid", type=int, default=None,
                    help="Shotid.")
parser.add_argument("-n", "--name", type=str, default="try_n", help="name")
args = parser.parse_args(sys.argv[1:])

basedir = "/work/05865/maja_n/stampede2/master/"


template = """likelihood:
    my_likelihood_class.StarLike:
      python_path: {}intensity-mapping/model/get_psf/integrated/
      stop_at_error: True
      shotid: {}
      input_params: [{} fwhm]

params:{}
  fwhm:
    prior:
      min: 1.0
      max: 4.0
    ref:
      min: {}
      max: {}

sampler:
  mcmc:
    Rminus1_stop: 0.03
    burn_in: 100
    max_tries: 1000000

output: {}cobaya-chains/psf_int_{}/{}"""

amp_temp = """
  A_{}:
    prior:
      min: 0
      max: {}
    ref:
      dist: norm
      loc: {}
      scale: {}"""
amp_str = ""
A_str = ""

# try to get the reported FWHM
cal_shot_tab = ascii.read(os.path.join(basedir, 'calshots_survey_table.tab'))
if args.shotid in cal_shot_tab['shotid']:
	shot_here = cal_shot_tab['shotid']==args.shotid
	fwhm_reported = cal_shot_tab['fwhm_virus'][shot_here].data[0]
	err_fwhm_reported = cal_shot_tab['fwhm_virus_err'][shot_here].data[0]
	fwhm_min = fwhm_reported - err_fwhm_reported
	fwhm_max = fwhm_reported + err_fwhm_reported
else:
	print(f'Shotid {args.shotid} is not in calibration shot list.')
	fwhm_min = 1.15
	fwhm_max = 1.35

stars = glob.glob(basedir+"radial_profiles/stars_{}/*.dat".format(args.shotid))
stars = np.sort(stars)

stars_good = open(basedir+"radial_profiles/stars_{}/use_stars.txt".format(args.shotid), "w")

i=0
for star in stars:
	a = ascii.read(star)
	rs = a["r"]
	order = np.argsort(rs)[:20]
	rs = rs[order]
	a = a[order]

	amax = np.nanmax(a["flux"])

	if amax < 2:
		continue
	
	amp_str += amp_temp.format(i, 3*amax, 1.5*amax, 0.4*amax)
	A_str += f"A_{i}, " 
	i+=1
	stars_good.write(star + "\n")
	
total_str = template.format(basedir, args.shotid, A_str, amp_str, fwhm_min, fwhm_max, basedir, args.name, args.name)

with open(args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open("cobaya_job.run", "w") as rf:
	rf.write("ibrun cobaya-run -f "+args.name+".yaml")
rf.close()
