import my_likelihood_plus as ml
import glob
import numpy as np
import sys

bello = ml.LaeLikePlus()
print("Initializing worked.")

print(bello.lae_ids)
print("getting the lae_ids worked")

print(bello.shot_ids)
print("number of shots: ", len(np.unique(bello.shot_ids)))
print("getting the shotid worked")

for shot in np.unique(bello.shot_ids):
	print(glob.glob(f"/work/05865/maja_n/stampede2/master/chains-laes/psf_{shot}"))

print("flus: ", bello.starflux.dtype)
print("err: ", bello.starsigma.dtype)
print("dist: ", bello.stardists.dtype)

kwargs = {}
for shot in bello.shot_ids:
	kwargs[f'fwhm_{shot}'] = 1.2

for laeid in bello.lae_ids:
	kwargs[f'Apow_{laeid}'] = 2
	kwargs[f'Apsf_{laeid}'] = 2
	

#for shot in bello.shot_ids:
#	kwargs[f"fwhm_{shot}"] = 1.5
#kwargs["A_pow"] = 1.0

print("likelihood test: ", bello.logp(**kwargs))
print("computing likelihood succeeded.")
