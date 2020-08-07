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
    my_likelihood_laes.MyLike:
      python_path: {}intensity-mapping/model/
      input_params: [{}]

params:{}

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
#dets_laes = dets_laes[dets_laes["shotid"]==args.shotid]

i=0
for lae_id in dets_laes:
	
	amax = 4.0
	amp_str += amp_temp.format(i, -1*amax, amax, 1.0)
	A_str += f"A_{i}, " 
	i+=1
	
A_str = A_str[:-2]
total_str = template.format(basedir, A_str, amp_str, basedir, args.name, args.name)

with open(args.name + ".yaml", "w") as yf:
	yf.write(total_str)
yf.close()

with open("cobaya_job.run", "w") as rf:
	rf.write("cobaya-run "+args.name+".yaml")
rf.close()
