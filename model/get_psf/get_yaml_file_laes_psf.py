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
    my_likelihood_class.LaeLike:
      python_path: {}intensity-mapping/model/get_psf/
      stop_at_error: True
      shotid: {}
      input_params: [fwhm, {}]

params:{}
  fwhm:
    prior:
      min: 1.0
      max: 3.0
    ref:
      dist: norm
      loc: {}
      scale: {}

prior:
  fwhm_prior: import_module("fwhm_prior").fwhm_prior

sampler:
  mcmc:
    Rminus1_stop: 0.001
    burn_in: 1
    max_tries: 1000000

output: {}chains-laes/psf_{}/{}"""

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
dets_laes = dets_laes[dets_laes["shotid"]==args.shotid]
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

fwhm_bf = ascii.read(basedir+"fwhm_posteriors/bf_fwhm.dat")
fwhm_loc = fwhm_bf["fwhm"][fwhm_bf["shotid"]==args.shotid].data[0]
fwhm_scale = 0.001

total_str = template.format(basedir, args.shotid, A_str, amp_str, fwhm_loc, fwhm_scale, basedir, args.name, args.name)

path = os.path.join(chaindir, "psf_"+args.name, "")
if not os.path.exists(path):
	os.mkdir(path)

with open(path+args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open(path+"cobaya_job.run", "w") as rf:
	rf.write("ibrun cobaya-run "+args.name+".yaml")
rf.close()
