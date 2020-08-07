import my_likelihood_lae as ml
import numpy as np

bello = ml.MyLike("ini.dataset")
print("Initializing worked.")

print(bello.lae_idx)
print("getting the lae_idx worked")

keys = [f"A_{i}" for i in range(bello.N_stars)]
kwargs = {}
for key in keys:
	kwargs[key] = 0.5

print("likelihood test: ", bello.logp(**kwargs))
print("computing likelihood succeeded.")
