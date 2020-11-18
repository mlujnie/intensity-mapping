import glob
from astropy.io import ascii
import numpy as np
import sys 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shotid", type=int, default=None,
                    help="Shotid.")
parser.add_argument("-n", "--name", type=str, default="try_n", help="name")
args = parser.parse_args(sys.argv[1:])

basedir = "/work/05865/maja_n/stampede2/master/"


template = """likelihood:
    my_likelihood_class.StarLike:
      python_path: {}intensity-mapping/model/get_psf/
      stop_at_error: True
      shotid: {}
      input_params: [{} fwhm]

params:{}
  fwhm:
    prior:
      min: 1.2
      max: 4.0
    ref:
      min: 1.2
      max: 2.0

sampler:
  mcmc:
    Rminus1_stop: 0.001
    burn_in: 1
    max_tries: 1000000

output: {}cobaya-chains/{}/{}"""

amp_temp = """
  A_{}:
    prior:
      min: 0
      max: {}
    ref:
      dist: norm
      loc: {}
      scale: 3"""
amp_str = ""
A_str = ""

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
	
	amp_str += amp_temp.format(i, 3*amax, amax)
	A_str += f"A_{i}, " 
	i+=1
	stars_good.write(star + "\n")
	
total_str = template.format(basedir, args.shotid, A_str, amp_str, basedir, args.name, args.name)

with open(args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open("cobaya_job.run", "w") as rf:
	rf.write("ibrun cobaya-run -f "+args.name+".yaml")
rf.close()
