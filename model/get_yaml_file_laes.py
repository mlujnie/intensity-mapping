import glob
from astropy.io import ascii
import numpy as np
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shotid", type=int, default=None,
                    help="Shotid.")
parser.add_argument("-n", "--name", type=str, default="try_n", help="name")
args = parser.parse_args(sys.argv[1:])

basedir = "/work/05865/maja_n/stampede2/master/"
chaindir = os.path.join(basedir, "chains-laes","")

template = """likelihood:
    my_likelihood_laes_hierarchical.HierLike:
      python_path: {}intensity-mapping/model/
      input_params: [mu_A, sigma_A, {}]

params:{}
  mu_A:
    prior:
      min: -10.0
      max: 10.0
    ref:
      min: 1.0
      max: 2.0
  sigma_A:
    prior:
      dist: expon
      loc: 0
      scale: 3.1
    ref:
      min: 2.5
      max: 3.5

sampler:
  mcmc:
    Rminus1_stop: 0.001
    burn_in: 1
    max_tries: 1000000

output: {}chains-laes/{}/{}"""

amp_temp = """
  A_{}:
    prior:
      min: {}
      max: {}
    ref:
      dist: norm
      loc: {}
      scale: 0.5"""
amp_str = ""
A_str = ""

dets_laes = ascii.read(basedir+"lists/dets_laes.tab")
dets_laes = dets_laes[dets_laes["vis_class"]>3]
dets_laes = dets_laes[np.argsort(dets_laes["detectid"])]

i=0
for lae_id in dets_laes:
	lae_idx = lae_id["detectid"]
	print(lae_idx)

	lae_path = "/work/05865/maja_n/stampede2/master/radial_profiles/laes/lae_{}.dat".format(lae_idx)
	if os.path.isfile(lae_path):
		amax = 20.0
		amp_str += amp_temp.format(lae_idx, -1*amax, amax, 1.0)
		A_str += f"A_{lae_idx}, "

A_str = A_str[:-2]
total_str = template.format(basedir, A_str, amp_str, basedir, args.name, args.name)

path = os.path.join(chaindir, args.name, "")
if not os.path.exists(path):
	os.mkdir(path)

with open(path+args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open(path+"cobaya_job.run", "w") as rf:
	rf.write("cobaya-run "+args.name+".yaml")
rf.close()
