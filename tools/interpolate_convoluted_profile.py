import numpy as np
from scipy.interpolate import RectBivariateSpline
import os

basedir = "/home/maja/Desktop/Master/Thesis/"

with open(os.path.join(basedir,"intensity-mapping/data/convolved_smoothed_powerlaw.dat"), "r") as cf:
    cfl = [x.strip().split(" ") for x in cf.readlines()]
    cfl = np.array(cfl, dtype=np.float)
    
fwhms = cfl[0]
conv_map = cfl[1:]

rdiff = 0.02
r = np.arange(2., 10, rdiff)

conv_func = RectBivariateSpline(fwhms, r[:-1]+rdiff/2., conv_map.T)

# check if it works
print(conv_func(1.3, np.arange(0,10,0.1))[0])
