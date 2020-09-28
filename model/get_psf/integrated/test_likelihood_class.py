import my_likelihood_class as ml
import numpy as np

bello = ml.LaeLike()

print(bello.stardists.shape)
print(bello.starflux.shape)
print(bello.starsigma.shape)

kwargs = {'fwhm' : 1.3}
for laeid in bello.lae_ids:
	kwargs['A_{}'.format(laeid)] = 2.0

print(bello.logp(**kwargs))
print('logp for LAEs worked.')

bello = ml.StarLike()

print(bello.stardists.shape)
print(bello.starflux.shape)
print(bello.starsigma.shape)

kwargs = {'fwhm' : 1.3}
for i in range(bello.N_stars):
	kwargs['A_{}'.format(i)] = 2.0

print(bello.logp(**kwargs))
print('logp for LAEs worked.')


