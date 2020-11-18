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
      python_path: {}intensity-mapping/model/psf_plus_halo/convolved/
      save_dir: /work/05865/maja_n/stampede2/master/radial_profiles/laes/
      input_params: [{}]
      stop_at_error: True

params:{}

sampler:
  mcmc:
    Rminus1_stop: 0.03
    burn_in: 1
    max_tries: 1000000

output: {}chains-laes/plus_{}/{}"""

amp_temp = """
  Apsf_{}: {}"""
amp_str = ""
A_str = ""

amp_pow_temp = """
  Apow_{}:
    prior:
      min: -10
      max: 10
    ref:
      min: -1
      max: 1"""

fwhm_temp = """
  fwhm_{}: {}"""
fwhm_str = ""
fwhm_pars = ""

dets_laes = ascii.read(basedir+"intensity-mapping/tables/new_laes.tab") #"lists/dets_laes.tab")
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
			amp_str += amp_pow_temp.format(lae_idx)
			A_str += f"Apsf_{lae_idx}, Apow_{lae_idx}, "

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

		print(shotid)
	except Exception as e:
		print("An error occurred with shotid "+str(shotid))
		print(e)
		pass

total_str = template.format(basedir, fwhm_pars + A_str, amp_str + fwhm_str, basedir, args.name, args.name)

path = os.path.join(chaindir, "plus_"+args.name, "")
if not os.path.exists(path):
	os.mkdir(path)

with open(path+args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open(path+"cobaya_job.run", "w") as rf:
	rf.write("ibrun cobaya-run -f "+args.name+".yaml\n")
rf.close()
