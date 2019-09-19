from multiprocessing import Pool
import multiprocessing
import numpy as np
import glob

from astropy.io import ascii, fits
from astropy.stats import biweight_scale

import time

import sys

# import hetdex stuff

# import my own utils : get_ffss, get_xrt_time
from tools_2d import *
from weighted_biweight import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, default="good_laes.txt", help="ascii table file containing the laes to use, with cat2==good.")
parser.add_argument("--cat", type=str, default="good", help="cat2 category to use")
parser.add_argument('--continuumcut', type=float, default=0., help='Exclude fibers with continuum > continuumcut [counts]. Currently turned off.')
parser.add_argument('--fwhmcut', type=float, default=0., help='Exclude all detections with a FWHM > fwhmcut [AA].')
parser.add_argument('--fluxcut', type=float, default=0., help='Exclude all detections with a lineflux > fluxcut [1e-17 ergs s cm2]')
parser.add_argument("-e", "--error", type=bool, default=False, help="Get random radial profiles.")
args = parser.parse_args(sys.argv[1:])

# count cpus

# define things
def_wave = np.arange(3470., 5542., 2.)

global_list = [ ] # len is going to be n, e.g. the number of shots

laetab = ascii.read(args.filename)
laetab = laetab[laetab["cat2"] == args.cat]
laetab = laetab[laetab["shotid"] != 20180110007] # this shot does not exist in multifits...

if args.fluxcut > 0.:
	laetab = laetab[laetab["flux"] <= args.fluxcut]
if args.fwhmcut > 0:
	laetab = laetab[laetab["fwhm"] <= args.fwhmcut/2.3548]

#print(f"\nUsing {len(laetab)} {args.cat} LAEs from {args.filename}\nlineflux <= {args.fluxcut}\nfwhm <= {args.fwhmcut}\n")

order = np.argsort(laetab["shotid"])
laetab = laetab[order]

shots_raw = np.unique(laetab["shotid"])
shots_raw = [str(x)[:-3]+"v"+str(x)[-3:] for x in shots_raw]
bads = ascii.read("badifus.dat")
bads = bads["shot"][bads["ifuamp"]=="alle"]

shotlist = []
for shot in shots_raw:
	if shot in bads:
		continue
	else:
		shotlist.append(shot)

shotlist = np.sort(shotlist)
distances = np.arange(0, 50., 3.)*1./3600.

def imfunc(idx):
	st = time.time()
	# get ffss, weights, etc
	shot = shotlist[idx]

	xrt = get_xrt_time(shot)
	ffss, ras, decs, thru, skysub, error, fibidx, amp, multis = get_ffss(shot)
	if type(ffss) == int:
		return 0
	ffss = np.concatenate(ffss)
	ras, decs = np.concatenate(ras), np.concatenate(decs)
	skysub = np.concatenate(skysub)
	error = np.concatenate(error)

	line = 3910
	here1 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	line = 4359
	here2 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	line = 5461
	here3 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	ffss[:,here1] = 0.
	ffss[:,here2] = 0.
	ffss[:,here3] = 0.

	ffss[ffss==0] = np.nan
	print('Finite ffss: ', ffss[np.isfinite(ffss)].size/ffss.size)
	weights = error**(-2.)
	weights[~np.isfinite(weights)] = 0.
	weights[~np.isfinite(ffss)] = 0.

	if args.continuumcut > 0:
		wlhere = (def_wave >= 4000) & (def_wave < 5000)
		continuum = np.nanmedian(ffss[:,wlhere], axis=1)
		print(f"\nBefore continuum cut of {args.continuumcut}: {ffss[np.isfinite(ffss)].size/ffss.size}")
		ffss[abs(continuum) > args.continuumcut] *= np.nan
		print(f"After continuum cut of {args.continuumcut}: {ffss[np.isfinite(ffss)].size/ffss.size}\n")

	imlist, weightlist = [[] for i in range(len(distances)-1)], [[] for i in range(len(distances)-1)]
	# loop through LAEs in this shot

	shotid = int(shot[:-4]+shot[-3:])
	tmptab = laetab[laetab["shotid"]==shotid]
	for lae_idx in range(len(tmptab)):
		#print(f"lae number {lae_idx} in {shot}")
		# define ra_lae and dec_lae etc.
		ifu, wave, ralae, declae = str(tmptab["ifuslot"][lae_idx]), tmptab["wl"][lae_idx], tmptab["ra"][lae_idx], tmptab["dec"][lae_idx]
		amp = tmptab["amp"][lae_idx]
	
		radiff, decdiff = ras - ralae, decs - declae
		diff = np.sqrt(radiff**2+decdiff**2)	
	
		wlhere = abs(def_wave - wave) <= 2.5

		# loop through distances and append fluxes to imlist, and weights to weightlist
		for counter, dist in enumerate(distances[:-1]):
			here = (diff > distances[counter]) & (diff <= distances[counter+1])
			for flux_0, weight_0 in zip(ffss[here], weights[here]):
				imlist[counter].append( flux_0[wlhere] )
				weightlist[counter].append( weight_0[wlhere] )

	try:	
		imlist = [np.concatenate(x) for x in imlist]
		weightlist = [np.concatenate(x) for x in weightlist]
	except ValueError:
		imlist_2 = []
		weightlist_2 = []
		for im, we  in zip(imlist, weightlist):
			if len(im) > 0:
				imlist_2.append(np.concatenate(im))
			else:
				imlist_2.append([])
			if len(we) > 0:
				weightlist_2.append(np.concatenate(we))
			else:
				weightlist_2.append([])
	
		imlist, weightlist = imlist_2, weightlist_2

	en = time.time()
	print("Time needed in loop: "+str(en-st))
	print("\nfinished {}.\n".format(idx))
	return imlist, weightlist # , skysubslist

cpu_count = multiprocessing.cpu_count()
print(cpu_count)
pool = Pool(12)#24)

n =  len(shotlist)
#print(f"len shotlist: {n}")
start = time.time()
endlist = pool.map(imfunc, np.arange(0, n)[::-1])
end = time.time()
print("Time needed: "+str(end-start))

print('len(endlist}, ', len(endlist))

imlists = [x[0] for x in endlist]
weightlists = [x[1] for x in endlist]
imlists = [[x[i] for x in imlists] for i in range(len(distances)-1)]
weightlists = [[x[i] for x in weightlists] for i in range(len(distances)-1)]
print('len(imlists) : ', len(imlists))

imlists = [np.concatenate(x) for x in imlists]
weightlists = [np.concatenate(x) for x in weightlists]

radial_biw = []
radial_maja = []
radial_karl = []
sigma = []
numbers = []
for counter in range(len(imlists)):
	fluxes, weights = imlists[counter], weightlists[counter]
	if len(fluxes) == 0:
		radial.append(0)
		sigma.append(0)
		continue

	radial_karl.append(biweight_location_weights_karl(fluxes, weights))
	radial_maja.append(biweight_location_weights(fluxes, weights))
	radial_biw.append(biweight_location(fluxes[np.isfinite(fluxes)]))

	N = len(fluxes[np.isfinite(fluxes)])
	numbers.append(N)
	std = biweight_scale(fluxes[np.isfinite(fluxes)])
	sigma.append(std)

#print(radial)
if args.error:
	ascii.write({"karl":radial_karl,"maja":radial_maja,"flux_biw": radial_biw, "sigma": sigma, "deltatheta":distances[:-1], 'number fibers':numbers}, "xi-2d-random-biw-weights-multi-karl-random-{}.dat".format(len(laetab)), overwrite=True)
else:
	ascii.write({"karl":radial_karl,"maja":radial_maja,"flux_biw": radial_biw, "sigma": sigma, "deltatheta":distances[:-1], 'number fibers':numbers}, "xi-2d-3A-{}-{}-{}-{}.dat".format(len(laetab), args.continuumcut, args.fluxcut, args.fwhmcut), overwrite=True)

