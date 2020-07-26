from multiprocessing import Pool
import multiprocessing
import numpy as np
from scipy.integrate import quad
import glob

from astropy.io import ascii, fits
from astropy.stats import biweight_scale
from astropy import units as u
from astropy.table import Table

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
parser.add_argument("--kpc", type=bool, default=False, help="Compute IM in proper kpc instead of degrees.")
args = parser.parse_args(sys.argv[1:])

def efunc(z, om, ol):
	return (om*(1+z)**3+ol)**(-1./2)

def integral(z, omega_m, omega_lambda):
	return quad(efunc, 0, z, args=(omega_m, omega_lambda))

def DA(z, h=0.674, omega_m=0.315, omega_lambda=0.685):
	Dh = 3000 * 1/h * u.Mpc 
	return Dh/(1+z)*integral(z, omega_m, omega_lambda)

def kpc_proper_per_deg(z, h=0.674, omega_m=0.315, omega_lambda=0.685):
	return np.pi/(648000)*3600*DA(z, h=0.674, omega_m=0.315, omega_lambda=0.685).to(u.kpc)


print("random: ", args.error)

# define things
def_wave = np.arange(3470., 5542., 2.)

global_list = [ ] # len is going to be n, e.g. the number of shots

laetab = ascii.read(args.filename)
#laetab = laetab[laetab["cat2"] == args.cat]
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
bads = ascii.read("/data/05865/maja_n/im2d/badifus.dat")
badshots = bads["shot"][bads["ifuamp"]=="alle"]

laenumber = 0
shotlist = []
for shot in shots_raw:
	if shot in badshots:
		continue
	else:
		shotlist.append(shot)
		lae = laetab[laetab["shotid"]==int(shot[:-4]+shot[-3:])]
		for i in range(len(lae)):
			ifu = str(lae["ifuslot"][i])
			if len(ifu)==2:
				ifu = "0"+ifu
			if ifu+"AA" in bads[bads["shot"]==shot]:
				continue
			else:
				laenumber += 1

shotlist = np.sort(shotlist)
distances = np.arange(0, 50., 3.)*1./3600.
if args.kpc:
	distances = kpc_proper_per_deg(2.5)[0]*distances
	print("distances in proper kpc: ", distances)

def imfunc(idx):
	st = time.time()
	# get ffss, weights, etc
	shot = shotlist[idx]

	try:
		xrt = get_xrt_time(shot)
		ffss, ras, decs, thru, skysub, error, fibidx, amp, multis, fluxcal = get_ffss(shot, asymcut=False)
	except Exception as e:
		print(e)
		print("In shot {}".format(shot))
		return 0
	if type(ffss) == int:
		return 0
	ffss = np.concatenate(ffss)
	ras, decs = np.concatenate(ras), np.concatenate(decs)
	fibidx = np.concatenate(fibidx)
	skysub = np.concatenate(skysub)
	error = np.concatenate(error)
	fluxcal = np.concatenate(fluxcal)
	amps = np.concatenate(amp)
	print("amps shape: ", amps.shape)
	ifuslots = np.concatenate([[x[10:13] for j in range(112)] for x in multis])
	print("ifuslots shape: ", ifuslots.shape)

	line = 3910
	here1 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	line = 4359
	here2 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	line = 5461
	here3 = np.where((def_wave>line-7)&(def_wave<line+7))[0]
	#ffss[:,here1] = 0.
	#ffss[:,here2] = 0.
	#ffss[:,here3] = 0.

	#ffss[:,:20] = 0.
	#ffss[:,-20:] = 0.

	ffss[ffss==0] = np.nan
	fluxcal[~np.isfinite(ffss)] = np.nan
	skysub[~np.isfinite(ffss)] = np.nan
	print('Finite ffss: ', ffss[np.isfinite(ffss)].size/ffss.size)
	weights = error**(-2.)
	weights[~np.isfinite(weights)] = 0.
	weights[~np.isfinite(ffss)] = 0.
	weights[weights>1.] = 0. # new !!!

	#print('min {}, med {}, max {}, std {}, biscale {}'.format(np.nanmin(weights),np.nanmedian(weights), np.nanmax(weights), np.nanstd(weights), biweight_scale(weights[np.isfinite(weights)])))

	if args.continuumcut > 0:
		wlhere = (def_wave >= 4000) & (def_wave < 5000)
		continuum = np.nanmedian(ffss[:,wlhere], axis=1)
		print(f"\nBefore continuum cut of {args.continuumcut}: {ffss[np.isfinite(ffss)].size/ffss.size}")
		ffss[abs(continuum) > args.continuumcut] *= np.nan
		print(f"After continuum cut of {args.continuumcut}: {ffss[np.isfinite(ffss)].size/ffss.size}\n")

	imlist, weightlist = [[] for i in range(len(distances)-1)], [[] for i in range(len(distances)-1)]
	callist =  [[] for i in range(len(distances)-1)]
	amplist =  [[] for i in range(len(distances)-1)]
	# loop through LAEs in this shot

	shotid = int(shot[:-4]+shot[-3:])
	tmptab = laetab[laetab["shotid"]==shotid]
	for lae_idx in range(len(tmptab)):
		#print(f"lae number {lae_idx} in {shot}")
		# define ra_lae and dec_lae etc.
		ifu = str(tmptab["ifuslot"][lae_idx])
		if len(ifu)==2:
			ifu = "0"+ifu
		if args.error:
			idxs = np.where(ifuslots==ifu)[0]
			print("len idxs: {}".format(len(idxs))) 
			#np.where(np.array([x[10:13] for x in multis]) == ifu)[0]			
			for iternum in range(20):
				diff0 = random.randint(0, len(idxs)-1)
				diff1 = idxs[diff0]
				lae_fibnum, ralae, declae = fibidx[diff1], ras[diff1], decs[diff1]
				wave = def_wave[random.randint(0, 1035)]
				zlae = wave/1215.67 - 1
			
				radiff, decdiff = ras - ralae, decs - declae
				diff = np.sqrt(radiff**2+decdiff**2)	
				if args.kpc:
					diff = kpc_proper_per_deg(zlae)[0]*diff 
			
				wlhere = abs(def_wave - wave) <=  2.5
				#longhere = (abs(def_wave - wave) <= 100. )&( ~wlhere)
				#print("len(longhere[longhere])", len(longhere[longhere]))

				# loop through distances and append fluxes to imlist, and weights to weightlist
				for counter, dist in enumerate(distances[:-1]):
					here = (diff > distances[counter]) & (diff <= distances[counter+1])
					for flux_0, weight_0, cal_0, amp_0 in zip(ffss[here], weights[here], fluxcal[here], skysub[here]):
						#tmplong = flux_0[longhere][np.isfinite(flux_0[longhere])]
						imlist[counter].append( flux_0[wlhere] )#- biweight_location(tmplong))
						weightlist[counter].append( weight_0[wlhere] )
						#tmpcallong = cal_0[longhere][np.isfinite(cal_0[longhere])]
						callist[counter].append(cal_0[wlhere] )#- biweight_location(tmpcallong))
						amplist[counter].append(amp_0[wlhere] )#- biweight_location(tmpcallong))


		else:
			ifu, wave, ralae, declae = str(tmptab["ifuslot"][lae_idx]), tmptab["wave"][lae_idx], tmptab["ra"][lae_idx], tmptab["dec"][lae_idx]
			amp = tmptab["amp"][lae_idx]
			zlae = wave/1215.67 - 1
	
			radiff, decdiff = ras - ralae, decs - declae
			diff = np.sqrt(radiff**2+decdiff**2)	
			if args.kpc:
				diff = kpc_proper_per_deg(zlae)[0]*diff 
		
			wlhere = abs(def_wave - wave) <= 2.5
			#longhere = (abs(def_wave - wave) <= 100. )&( ~wlhere)
			#print("len(longhere[longhere])", len(longhere[longhere]))


			# loop through distances and append fluxes to imlist, and weights to weightlist
			for counter, dist in enumerate(distances[:-1]):
				here = (diff > distances[counter]) & (diff <= distances[counter+1])
				for flux_0, weight_0, cal_0, amp_0 in zip(ffss[here], weights[here], fluxcal[here], skysub[here]):
					#tmplong = flux_0[longhere][np.isfinite(flux_0[longhere])]
					imlist[counter].append( flux_0[wlhere] )#-biweight_location(tmplong))
					weightlist[counter].append( weight_0[wlhere] )
					#tmpcallong = cal_0[longhere][np.isfinite(cal_0[longhere])]
					callist[counter].append(cal_0[wlhere] )#- biweight_location(tmpcallong))
					amplist[counter].append(amp_0[wlhere] )#- biweight_location(tmpcallong))

	try:	
		imlist = [np.concatenate(x) for x in imlist]
		weightlist = [np.concatenate(x) for x in weightlist]
		callist = [np.concatenate(x) for x in callist]
		amplist = [np.concatenate(x) for x in amplist]
	except ValueError:
		imlist_2 = []
		weightlist_2 = []
		callist_2 = []
		amplist_2 = []
		for im, we, ca, am  in zip(imlist, weightlist, callist, amplist):
			if len(im) > 0:
				imlist_2.append(np.concatenate(im))
			else:
				imlist_2.append([])
			if len(we) > 0:
				weightlist_2.append(np.concatenate(we))
			else:
				weightlist_2.append([])
			if len(ca) > 0:
				callist_2.append(np.concatenate(ca))
			else:
				callist_2.append([])
			if len(am) > 0:
				amplist_2.append(np.concatenate(am))
			else:
				amplist_2.append([])
	
		imlist, weightlist = imlist_2, weightlist_2
		callist = callist_2
		amplist = amplist_2

	try:	
		radial_biw = []
		radial_maja = []
		radial_karl = []
		cal_biw = []
		cal_karl = []
		amp_karl = []
		sigma = []
		numbers = []
		median = []	
		for x in range(len(imlist)):
			if len(imlist[x])>0:
				radial_biw.append(biweight_location(imlist[x][np.isfinite(imlist[x])]))
				radial_karl.append(biweight_location_weights_karl(imlist[x], weightlist[x]))
				radial_maja.append(biweight_location_weights(imlist[x], weightlist[x]))
				cal_biw.append(biweight_location(callist[x][np.isfinite(callist[x])]))
				cal_karl.append(biweight_location_weights_karl(callist[x], weightlist[x]))
				sigma.append(biweight_scale(imlist[x][np.isfinite(imlist[x])]))
				numbers.append(len(imlist[x]))
				median.append(np.nanmedian(imlist[x]))
				amp_karl.append(biweight_location_weights_karl(amplist[x], weightlist[x]))
			else:
				radial_biw.append(0)
				radial_karl.append(0)
				radial_maja.append(0)
				cal_biw.append(0)
				cal_karl.append(0)
				sigma.append(0)
				numbers.append(0)
				median.append(0)
				amp_karl.append(0)

		if args.kpc:
			dist_string="delta_r[kpc]"
		else:
			dist_string="deltatheta"
		#ascii.write({"karl":radial_karl,"maja":radial_maja,"flux_biw": radial_biw, "sigma": sigma, dist_string:distances[:-1], 'number fibers':numbers, 'median':median, "cal_biw":cal_biw, "cal_karl":cal_karl, "amp_karl":amp_karl}, "radials_sub/{}.dat".format(idx), overwrite=True)
	except Exception as e:
		print("Error: {}".format(e))
		pass

	en = time.time()
	print("Time needed in loop: "+str(en-st))
	print("\nfinished {}.\n".format(idx))
	return imlist, weightlist , callist, amplist

cpu_count = multiprocessing.cpu_count()
print(cpu_count)
pool = Pool(6)

n =  len(shotlist)

#print(f"len shotlist: {n}")
start = time.time()
endlist = pool.map(imfunc, np.arange(0, n))
end = time.time()
print("Time needed: "+str(end-start))

finallist = []
for x in endlist:
	if type(x) != int:
		finallist.append(x)
		

print('len(endlist}, ', len(finallist))

imlists = [x[0] for x in finallist]
weightlists = [x[1] for x in finallist]
callists = [x[2] for x in finallist]
amplists = [x[3] for x in finallist]
imlists = [[x[i] for x in imlists] for i in range(len(distances)-1)]
weightlists = [[x[i] for x in weightlists] for i in range(len(distances)-1)]
callists = [[x[i] for x in callists] for i in range(len(distances)-1)]
amplists = [[x[i] for x in amplists] for i in range(len(distances)-1)]
print('len(imlists) : ', len(imlists))

imlists = [np.concatenate(x) for x in imlists]
weightlists = [np.concatenate(x) for x in weightlists]
callists = [np.concatenate(x) for x in callists]
amplists = [np.concatenate(x) for x in amplists]

radial_biw = []
radial_maja = []
radial_karl = []
cal_biw = []
cal_karl = []
amp_karl = []
amp_sigma = []
sigma = []
numbers = []
median = []
cal_sigma = []
for counter in range(len(imlists)):
	fluxes, weights = imlists[counter], weightlists[counter]
	cals = callists[counter]
	ampws = amplists[counter]
	if len(fluxes) == 0:
		radial.append(0)
		sigma.append(0)
		continue

	radial_karl.append(biweight_location_weights_karl(fluxes, weights))
	radial_maja.append(biweight_location_weights(fluxes, weights))
	radial_biw.append(biweight_location(fluxes[np.isfinite(fluxes)]))
	median.append(np.nanmedian(fluxes))
	cal_biw.append(biweight_location(cals[np.isfinite(cals)]))
	cal_karl.append(biweight_location_weights_karl(cals, weights))
	cal_sigma.append(biweight_scale(cals[np.isfinite(cals)]))
	amp_karl.append(biweight_location_weights_karl(ampws, weights))
	amp_sigma.append(biweight_scale(ampws[np.isfinite(ampws)]))

	N = len(fluxes[np.isfinite(fluxes)])
	numbers.append(N)
	std = biweight_scale(fluxes[np.isfinite(fluxes)])
	sigma.append(std)


dist_string = "deltatheta"
rstring = "deg"
if args.kpc:
	dist_string = "delta_r[kpc]"
	rstring = "kpc"

metadict = { "comments":["dist = "+rstring,
		"LAE number = {}".format(laenumber),
		"continuum cut = ".format(args.continuumcut),
		"flux cut = {}".format(args.fluxcut),
		"fwhm cut = {}".format(args.fwhmcut)]}
if args.error:
	metadict["error"] = "True"
else:
	metadict["error"] = "False"

outtable = Table({"karl":radial_karl,"maja":radial_maja,"flux_biw": radial_biw, "sigma": sigma, dist_string:distances[:-1], 'number fibers':numbers, 'median':median, "cal_biw":cal_biw, "cal_karl":cal_karl, "cal_sigma":cal_sigma, "amp_karl":amp_karl, "amp_sigma":amp_sigma}, meta = metadict)

if args.error:
	ascii.write(outtable, "xi-{}-3A-{}-{}-{}-{}-rand-weightcutoff-symcut5-lessedge-wamp.dat".format(rstring, laenumber, args.continuumcut, args.fluxcut, args.fwhmcut), overwrite=True)
else:
	ascii.write(outtable, "xi-{}-3A-{}-{}-{}-{}-weightcutoff-symcut5-lessedge-wamp.dat".format(rstring, laenumber, args.continuumcut, args.fluxcut, args.fwhmcut), overwrite=True)

