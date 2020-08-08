import my_likelihood_lae as ml
import numpy as np
import sys

bello = ml.MyLike()
print("Initializing worked.")
sys.exit()

print(bello.lae_ids)
print("getting the lae_ids worked")

keys = [f"A_{i}" for i in bello.lae_ids]
kwargs = {}
for key in keys:
	kwargs[key] = 0.5
kwargs["mu_A"] = 1.7
kwargs["sigma_A"] = 3.0

print("likelihood test: ", bello.logp(**kwargs))
print("computing likelihood succeeded.")
