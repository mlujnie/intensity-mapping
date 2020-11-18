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
    my_likelihood_plus.LaeLikePlus:
      python_path: {}intensity-mapping/model/
      input_params: [A_pow, {}]
      stop_at_error: True

params:{}
  A_pow:
    prior:
      min: -10
      max: 10
    ref:
      dist: norm
      loc: 0
      scale: 3

prior:{}

sampler:
  mcmc:
    Rminus1_stop: 0.001
    burn_in: 1
    max_tries: 1000000

output: {}chains-laes/plus_{}/{}"""

amp_temp = """
  Apsf_{}: {}"""
amp_str = ""
A_str = ""

fwhm_temp = """
  fwhm_{}: {}"""
fwhm_str = ""
fwhm_pars = ""

prior_amp_temp = """
  Apsf_{}_prior: import_module('external_priors').Apsf_{}_prior"""
prior_fwhm_temp = """
  fwhm_{}_prior: import_module('external_priors').fwhm_{}_prior"""

prior_str = ""

dets_laes = ascii.read(basedir+"lists/dets_laes.tab")
dets_laes = dets_laes[dets_laes["vis_class"]>3]
dets_laes = dets_laes[np.argsort(dets_laes["detectid"])]

i=0
shots = []
for lae_id in dets_laes:
	lae_idx = lae_id["detectid"]

	lae_path = "/work/05865/maja_n/stampede2/master/radial_profiles/laes/lae_{}.dat".format(lae_idx)
	A_bf = ascii.read(basedir+"lae_psf_posterior/bf_amp.dat")
	if os.path.isfile(lae_path):

		try:
			amp_loc = A_bf["A"][A_bf["detectid"]==lae_idx].data[0]
			amp_str += amp_temp.format(lae_idx, amp_loc)
			A_str += f"Apsf_{lae_idx}, "

			prior_str += prior_amp_temp.format(lae_idx, lae_idx)

			shots.append(lae_id["shotid"])
		except Exception as e:
			print(f"LAE {lae_idx} failed.")
			print(e)

A_str = A_str[:-2]

for shotid in np.unique(shots):
	try: 
		fwhm_bf = ascii.read(basedir+"fwhm_posteriors/bf_fwhm.dat")
		fwhm_loc = fwhm_bf["fwhm"][fwhm_bf["shotid"]==shotid].data[0]
		fwhm_scale = 0.001

		fwhm_str += fwhm_temp.format( shotid, fwhm_loc)
		fwhm_pars += f"fwhm_{shotid}, "

		prior_str += prior_fwhm_temp.format(shotid, shotid)

		print(shotid)
	except Exception as e:
		print("An error occurred with shotid "+str(shotid))
		print(e)
		pass

total_str = template.format(basedir, fwhm_pars + A_str, amp_str + fwhm_str, prior_str, basedir, args.name, args.name)

path = os.path.join(chaindir, "plus_"+args.name, "")
if not os.path.exists(path):
	os.mkdir(path)

with open(path+args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open(path+"cobaya_job.run", "w") as rf:
	rf.write("ibrun cobaya-run -f "+args.name+".yaml\n")
rf.close()
