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
print(chaindir)

template = """likelihood:
    my_likelihood_class.LaeLike:
      python_path: {}intensity-mapping/model/get_psf/integrated/
      stop_at_error: True
      shotid: {}
      input_params: [fwhm, {}]

params:{}
  fwhm:
    prior:
      min: 1.0
      max: 2.0
    ref:
      dist: norm
      loc: {}
      scale: {}

prior:
  fwhm_prior: import_module("fwhm_prior").fwhm_prior

sampler:
  mcmc:
    Rminus1_stop: 0.03
    burn_in: 1
    max_tries: 1000000

output: {}chains-laes/psf_int_{}/{}"""

amp_temp = """
  A_{}:
    prior:
      min: {}
      max: {}
    ref:
      dist: norm
      loc: {}
      scale: {}"""
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
		tmp = ascii.read(lae_path)
		tmp = tmp[tmp['r']<5]
		amax = np.nanmax(tmp['flux'])
		amp_str += amp_temp.format(lae_idx, 0, 4*amax, 1.5*amax, 0.4*amax)
		A_str += f"A_{lae_idx}, "

A_str = A_str[:-2]

fwhm_bf = ascii.read(basedir+"fwhm_posteriors/integrated/bf_fwhm.dat") # will have to change this when I have the new FWHMs
fwhm_loc = fwhm_bf["fwhm"][fwhm_bf["shotid"]==args.shotid].data[0]
fwhm_scale = 0.001

total_str = template.format(basedir, args.shotid, A_str, amp_str, fwhm_loc, fwhm_scale, basedir, args.name, args.name)

path = os.path.join(chaindir, "psf_int_"+args.name, "")
if not os.path.exists(path):
	os.mkdir(path)

with open(path+args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open(path+"cobaya_job.run", "w") as rf:
	rf.write("cobaya-run -f "+args.name+".yaml")
rf.close()
