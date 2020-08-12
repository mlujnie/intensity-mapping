import my_likelihood_class as ml
import numpy as np
import sys

bello = ml.LaeLike()
print("Initializing worked.")

print(bello.lae_ids)
print("getting the lae_ids worked")

print(bello.shotid)
print("getting the shotid worked")

print("flus: ", bello.starflux.dtype)
print("err: ", bello.starsigma.dtype)
print("dist: ", bello.stardists.dtype)

keys = [f"A_{i}" for i in bello.lae_ids]
kwargs = {}
for key in keys:
	kwargs[key] = 0.5
kwargs["fwhm"] = 1.6

print("likelihood test: ", bello.logp(**kwargs))
print("computing likelihood succeeded.")
