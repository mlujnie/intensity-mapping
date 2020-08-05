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


template = """likelihood:
    my_likelihood_class.MyLike:
      python_path: /data/05865/maja_n/intensity-mapping/model/
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

output: /data/05865/maja_n/cobaya-chains/{}/{}"""

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

stars = glob.glob("/data/05865/maja_n/radial_profiles/stars_{}/*".format(args.shotid))
stars = np.sort(stars)

for i, star in enumerate(stars):
	a = ascii.read(star)
	amax = np.nanmax(a["flux"])
	amp_str += amp_temp.format(i, 3*amax, amax)
	A_str += f"A_{i}, " 
	
total_str = template.format(A_str, amp_str, args.name, args.name)

with open(args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open("cobaya_job.run", "w") as rf:
	rf.write("cobaya-run "+args.name+".yaml")
rf.close()
