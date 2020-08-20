import my_likelihood_plus as ml
import numpy as np
import sys

bello = ml.LaeLikePlus()
print("Initializing worked.")

print(bello.lae_ids)
print("getting the lae_ids worked")

print(bello.shot_ids)
print("getting the shotid worked")

print("flus: ", bello.starflux.dtype)
print("err: ", bello.starsigma.dtype)
print("dist: ", bello.stardists.dtype)

keys = [f"Apsf_{i}" for i in bello.lae_ids]
kwargs = {}
for key in keys:
	kwargs[key] = 0.5

for shot in bello.shot_ids:
	kwargs[f"fwhm_{shot}"] = 1.5
kwargs["A_pow"] = 1.0

print("likelihood test: ", bello.logp(**kwargs))
print("computing likelihood succeeded.")
