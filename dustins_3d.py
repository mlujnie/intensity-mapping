import matplotlib
import random
#matplotlib.use("Agg")
from astropy.stats import biweight_scale, biweight_location
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from scipy.ndimage.filters import gaussian_filter
import tables as tb
import glob
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pickle
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--error", type=bool, default=False,
                    help="Get random radial profiles.")
parser.add_argument("-k", "--kappa", type=float, default=1.7, help="Kappa for continuum source flagging.")
args = parser.parse_args(sys.argv[1:])

def get_closest_date(inpath, shot): 
	pp = np.sort(glob.glob(inpath))
	date = int(shot[:-4])
	dates_inpath = np.array([int(x.split("/")[-1]) for x in pp])
	date_diff = date - dates_inpath
	out_date_idx = np.where(date_diff>=0)[0][-1]
	out_date_path = pp[out_date_idx]
	return out_date_path


def get_xrt_time_new(shot, ifu):
	inpath = "/work/05865/maja_n/stampede2/midratio/*"
	outpath = get_closest_date(inpath, shot)
	pattern = outpath+"/{}.dat"
	xrt = {}
	SIGMA = 4.
	#weirdampslist = [[('035','LL'),590, 615],[('082','RL'),654,681],[('023','RL'), 349, 376],[('026','LL'), 95,142]]
	multinames = [x.split('/')[-1][:-4] for x in np.unique(glob.glob(pattern.format('multi_???_'+ifu+'_???_??')))]
	#for amp in weirdampslist:
	#		key, start, stop = amp
	#		xrt_0[key] = np.concatenate([xrt_0[key][:start],np.interp(np.arange(stop-start),[0,stop-start],[xrt_0[key][start],xrt_0[key][stop]]),xrt_0[key][stop:]])
	wave = def_wave
	line = 3910
	here1 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 4359
	here2 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 5461
	here3 = np.where((wave>line-10)&(wave<line+10))[0]
	SMOOTHATA = True
	if SMOOTHATA:
			for multi in multinames:
					key = (multi[10:13], multi[18:20])
					tmp = ascii.read(pattern.format( multi))
					wl, xrt_0 = tmp["wl"], tmp["midratio"]
					here = here1
					slope = (xrt_0[here[-1]+1] - xrt_0[here[0]-1])/float(len(here))
					xrt_1 = np.concatenate([xrt_0[:here[0]], xrt_0[here[0]-1] + np.arange(len(here))*slope, xrt_0[here[-1]+1:]])
					here = here2
					slope = (xrt_0[here[-1]+1] - xrt_0[here[0]-1])/float(len(here))
					xrt_1 = np.concatenate([xrt_0[:here[0]], xrt_0[here[0]-1] + np.arange(len(here))*slope, xrt_0[here[-1]+1:]])
					here = here3
					slope = (xrt_0[here[-1]+1] - xrt_0[here[0]-1])/float(len(here))
					xrt_1 = np.concatenate([xrt_0[:here[0]], xrt_0[here[0]-1] + np.arange(len(here))*slope, xrt_0[here[-1]+1:]])
					xrt_1 = np.interp(np.arange(len(xrt_1)), np.arange(len(xrt_1))[np.isfinite(xrt_1)], xrt_1[np.isfinite(xrt_1)])
					#print xrt_1[~np.isfinite(xrt_1)]
					xrt[key] = interp1d(wave, gaussian_filter(xrt_1, sigma=SIGMA/2.), fill_value=(xrt_1[0],xrt_1[-1]),bounds_error=False)
	else:
			for key in xrt_0.keys():
					xrt[key] = interp1d(wave, xrt_0[key], fill_value=(xrt_0[key][0],xrt_0[key][-1]),bounds_error=False)
	return xrt

def load_shots(shotlist, ifu):
	tables = {}	
	for shot in shotlist:#[shot]: #shotlist:
		filename = '/work/03946/hetdex/hdr1/reduction/data/{}.h5'.format(shot)
		fileh = tb.open_file(filename, 'r')

		table = Table(fileh.root.Data.Fibers.read_where("ifuslot==ifu"))
		#tables.append(table)
		tables[shot] = table
		midra, middec =  np.median(table["ra"]), np.median(table["dec"])
		print "midra, middec = ", np.median(table["ra"]), np.median(table["dec"])
		fileh.close()
		print "\n"
	return tables, midra, middec

def flag_next(flag):
	newflagged = []
	for i in range(len(flag)):
		if not flag[i]:
			if i in newflagged:
				continue
			else:
				if i%112==0& i!=0 & flag[i+1]:
					flag[i+1] = False
					newflagged.append(i+1)
				elif i%112==111 & flag[i-1]:
					flag[i-1] = False
					newflagged.append(i-1)
				elif i == 0 :
					if flag[i+1]:
						flag[i+1] = False
						newflagged.append(i+1)
				elif i!=flag.size-1:
					if flag[i-1]:
						flag[i-1] = False
						newflagged.append(i-1)
					if flag[i+1]:
						flag[i+1] = False
						newflagged.append(i+1)
	return flag


badamplist = {("024","LL"):np.zeros((112,1036)),
                         ("024","LU"):np.zeros((112,1036)),
                         ("024","RL"):np.zeros((112,1036)),
                         ("024","RU"):np.zeros((112,1036)),
                         ("032","LU"):np.zeros((112,1036)),
                         ("032","LL"):np.zeros((112,1036)),
                         ("083","RU"):np.zeros((112,1036)),
                         ("083","RL"):np.zeros((112,1036)),
                         ("046","RU"):np.zeros((112,1036)),
                         ("046","RL"):np.zeros((112,1036)),
                         ("092","LL"):np.zeros((112,1036)),
                         ("092","LU"):np.zeros((112,1036)),
                         ("092","RL"):np.zeros((112,1036)),
                         ("092","RU"):np.zeros((112,1036)),
                         ("095","RU"):np.zeros((112,1036)),
        #                 ("096","LL"):etwas,
                         ("106","RU"):np.zeros((112,1036))}

def get_ffss(shot, ifu):
        ff = []
        shots = []
        table = tables[shot]
        
        ff = glob.glob("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_{}_???_??.fits".format(shot,ifu))
	alle = glob.glob("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_???_???_??.fits".format(shot))
	
	multis = []
	fibers = []
	exps = []
	for counter, fin in enumerate(alle):
		multi = fin.split("/")[-1][:-5]
		exps.append(fin.split("/")[-2])
		multis.append(multi)	
		fibers.append(fits.open(fin)[0].data)
		if (multi[10:13], multi[18:20]) in badamplist.keys():
			fibers[counter] *= badamplist[(multi[10:13], multi[18:20])]
	
	exps = np.array(exps)
	multis = np.array(multis)
	fibers = np.array(fibers)
	fibers[fibers==0] = np.nan
	meanmean = np.nanmedian(np.concatenate(fibers), axis=1)
	sorted_mean = np.argsort(meanmean)
	finites = meanmean[np.isfinite(meanmean)].size
	maxim = meanmean[sorted_mean[int(finites*0.94)]]
	flag = meanmean <= maxim
	#print "flagged: ", flag[flag].size/float(flag.size)
	#flag = flag_next(flag)
	#print "flagged now: ", flag[flag].size/float(flag.size)

	flag = np.array(np.split(flag, len(alle)))
	
	idx = [x[10:13]==ifu for x in multis]
	exps = exps[idx]
	multis = multis[idx]
	amps = np.array([x[18:20] for x in multis])
	flag = flag[idx]
	fibers = fibers[idx]
	idx = None

        multiname = []
        ffss = []
        ras, decs = [], []
        throughputs = []
        skysubs = []
	errors = []
	fibnums = []
        #for i, fin in enumerate(ff):
	for i in range(len(multis)):
		amp = amps[i]
		idx = (table["multiframe"]==multis[i])&(table["expnum"]==int(exps[i][-1]))
                if len(idx[idx])==0:
                        continue

		a = fibers[i]
		a[~flag[i]] = 0.
                
		f2f = np.array([np.interp(def_wave, x, y) for x, y in zip(table["wavelength"][idx], table["fiber_to_fiber"][idx])])
                throughput = table["Throughput"][idx]

                A2A = table["Amp2Amp"][idx]
                a2a = np.ones(A2A.shape)
                a2a[A2A == 0 ] = 0

                ra, dec = table["ra"][idx], table["dec"][idx]

		skysub = []
		for x, y in zip(table["wavelength"][idx], table["sky_subtracted"][idx]):
			try:
				skysub.append(np.interp(def_wave, x[y!=0], y[y!=0], left=0.0, right=0.0))
			except ValueError:
				skysub.append(np.zeros(def_wave.shape))
		#try:
               # 	skysub  =  np.array([np.interp(def_wave, x[y!=0], y[y!=0], left=0.0, right=0.0) for x, y in zip(table["wavelength"][idx], table["sky_subtracted"][idx])])
		#except Exception as e:
		#	print e
		#	print "ERROR IS HERE!"
		#A	skysub = np.zeros((112, 2036))
		skysub = np.array(skysub)
		skysub[~flag[i]] = 0.0
                skysubs.append(skysub/(f2f*xrt[(ifu, amp)](def_wave))*a2a)
                
		error  =  np.array([np.interp(def_wave, x, y, left=0.0, right=0.0) for x, y in zip(table["wavelength"][idx], table["error1Dfib"][idx])])
                errors.append(error)
		
		fibnums.append(table["fibidx"][idx])

                ffss.append(a/(f2f*xrt[(ifu, amp)](def_wave))*a2a)
                ras.append(ra)
                decs.append(dec)
                throughputs.append(throughput)
		#amps.append(table["amp"][idx])
                #print ra.shape, dec.shape
	amps = [[x for i in range(112)] for x in amps]
        ffss = np.array(ffss)
        return ffss, np.array(ras), np.array(decs), np.array(throughputs), np.array(skysubs), np.array(errors), np.array(fibnums), np.array(amps)

def gaus(x, a, b, c):
        return a*np.exp(-(x-b)**2/(2*c**2))

def dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def isinap(a, b, inner, outer):
        distance = dist(a,b)
        return (distance<outer)&(distance>inner)

def get_fluxmap(midwave, midcoord, radecs, ffss, weights, sigma, linewidth=None):

	ap0 = 5./3600
	a = midcoord # (ra, dec)
	radial0 = []
	for i in range(len(ffss)):
		b = radecs[i]
		if isinap(a, b, 0.0, ap0):
			radial0.append(ffss[i])
	
	radial0 = np.nanmedian(radial0, axis=0)
	idx = abs(def_wave-midwave) <= 30.
	if linewidth==None:
		try:
			popt, pcov = curve_fit(gaus, def_wave[idx], radial0[idx], p0=[5., midwave, 3.])
			linewidth = 2.355*popt[2]/2.
		except RuntimeError as e:
			print e
			linewidth = 10.
	print "linewidth ", linewidth
	#ragrid, decgrid = np.arange(np.nanmin(ras)-1./3600, np.nanmax(ras)+1./3600, 1./3600), np.arange(np.nanmin(decs)-1./3600, np.nanmax(decs)+1./3600, 1./3600)

	fluxmap = np.zeros((decgrid.size,ragrid.size))

	idx, fibidx, multiframe, expnum = {},{},{},{}

	for i in range(fluxmap.shape[0]):
		for j in range(fluxmap.shape[1]):

			fibidx = np.where((ras<=ragrid[j]+1/3600.)&(ras>=ragrid[j]-0/3600.)&(decs<=decgrid[i]+1./3600.)&(decs>=decgrid[i]-0./3600.))[0]
			if len(fibidx) == 0:
				etwas = [0]
				etwas2 = [1]
			else:
				etwas = []
				etwas2 = []
				for l in range(len(fibidx)):
					tmp = ffss[fibidx[l]]*weights[fibidx[l]]     #[(shots==shot)&(multiname==multiframe[(shot,ap)][l])&(exps==expnum[(shot,ap)][l])][0][fibidx[(shot,ap)][l]]
					tmp[tmp==0] = np.nan
					etwas.append(tmp[(def_wave<=midwave+5.) & (def_wave>=midwave-5.)]) 
					etwas2.append(weights[fibidx[l]][(def_wave<=midwave+5.) & (def_wave>=midwave-5.)])
			fluxmap[i,j] = np.nansum(etwas)/np.nansum(etwas2) #np.nanmean(etwas)

	return gaussian_filter(fluxmap, sigma = sigma)




shotlist_D = """20181215v031
20190108v013
20190112v023""".split("\n")

shotlist_C = """20181118v020
20181119v015
20181119v016
20181120v013
20181120v012""".split("\n")

#shotlist = shotlist_C
def_wave = np.arange(3470., 5542., 2.)
#xrt = get_xrt_time_new() #get_xrt_new()

#detects_cosmos = ascii.read("detects_comsos_3.dat") #detects_comsos_2.dat ifuslot, wl, ra, dec
#detects_cosmos = ascii.read("COSMOSC_halo_cand.cat")  # ifu, wl_com, ra_com, dec_com
detects_cosmos = ascii.read("dustins-laes-2.dat") # ifuslot dec ra wl shotid

#ifus = detects_cosmos["ifuslot"]
#waves = detects_cosmos["wl"]
#radecs = zip(detects_cosmos["ra"], detects_cosmos["dec"])

allgood = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 25, 27, 28, 29, 30, 32, 34, 35, 36, 39, 40, 41, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 118, 119, 121, 122, 124, 125, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 148, 149, 150, 151, 152, 154, 156, 158, 160, 161, 162, 163, 165, 166, 167, 168, 170, 172, 174, 176, 178, 179, 181, 184, 185, 187, 189, 190, 191, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209, 210, 212, 213, 214, 215]

detects_19 = ascii.read("goodlaes-19.dat")
good_19 = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 51, 54, 55, 57, 60, 63, 64, 66, 67, 69, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 84, 89, 90, 91, 92, 93]
detects_19 = detects_19[good_19]
print "number of LAEs in 19 ", len(detects_19["ifuslot"])

detects_cosmos = detects_cosmos[detects_cosmos["shotid"]>20180000000]
#print "number of laes: ", len(detects_cosmos["shotid"])
detects_cosmos = detects_cosmos[allgood]
print "number of laes: ", len(detects_cosmos["shotid"])

joined = vstack([detects_cosmos, detects_19])
print "total number of LAEs: ", len(joined)

######################### change this later ################################
number = {}
for ifu in joined['ifuslot']:
	try:
		number[ifu] += 1
	except KeyError:
		number[ifu] = 1
for key in number.keys():
	print key, ' : ', number[key]
#############################################################################


global_ragrid, global_decgrid = np.arange(-30, 31, 1)*1/3600., np.arange(-30, 31, 1)*1/3600.

MAXAP = 50.
apertures = np.logspace(np.log10(1./3600), np.log10(MAXAP/3600.), 20)

distances = np.logspace(np.log10(40.), np.log10(1400.), 20)
print "distances",  distances
zs = def_wave/1215.67 - 1
print zs

######### redshifts
redshifts = joined['wl']/1215.67 - 1.
print redshifts.min(), redshifts.max()
redshift_bins = [1.9, 2.43, 2.96, 3.5]
bin_idx = []
for counter, maxz in enumerate(redshift_bins[1:]):
	here = np.where((redshifts>=redshift_bins[counter])&(redshifts<maxz))[0]
	bin_idx.append(here)
print bin_idx


distances_here = cosmo.comoving_distance(zs)
print "distances_here", distances_here

#medfile = open("median_values_{}.dat".format(args.kappa), "w")
#medfile.write("ffskysub"+"	"+"original"+"\n")

#before = open("before2.dat", "w")
#before.write("ffskysub"+"\t"+"original"+"\n")

fluxmaps = []
radial_fibers = [[] for i in range(len(apertures)-1)]
radial_weights = [[] for i in range(len(apertures)-1)]
radial_skysubs = [[] for i in range(len(apertures)-1)]
for i in range(len(joined["ifuslot"])):
	try:
		ifu, wave, (ralae, declae), shot = str(joined["ifuslot"][i]), joined["wl"][i], (joined["ra"][i], joined["dec"][i]), str(joined["shotid"][i])
		amp = joined["amp"][i]
		shotlist = [shot[:-3]+"v"+shot[-3:]]
		lae_fibnum = joined["fibnum"][i]
		if len(ifu)==2:
			ifu = "0"+ifu
		#print ifu, wave, ralae, declae

		xrt = get_xrt_time_new(shot[:-3]+"v"+shot[-3:], ifu) #get_xrt_new()
		tables, midra, middec = load_shots(shotlist, ifu)

		ffss, ras, decs = [], [], []
		thrus = []
		skysubs = []
		amps = []
		errorframe = []
		fibnums = []
		#print shotlist
		for shot in shotlist:
 			ffss0, ras0, decs0, thru0, skysub, error, fibidx, amp0 = get_ffss(shot, ifu)
			ffss.append(ffss0)
			ras.append( ras0)
			decs.append(decs0)
			thrus.append(thru0)
			skysubs.append(skysub)
			errorframe.append(error)
			fibnums.append(fibidx+1)
			amps.append(amp0)
    
		try:
			ffss, ras, decs = np.concatenate(np.concatenate(ffss)), np.concatenate(np.concatenate(ras)), np.concatenate(np.concatenate(decs))
			thrus = np.concatenate(np.concatenate(thrus))
			skysubs = np.concatenate(np.concatenate(skysubs))
			errorframe = np.concatenate(np.concatenate(errorframe))
			fibnums = np.concatenate(np.concatenate(fibnums))
			amps = np.concatenate(np.concatenate(amps))
		except ValueError as e:
			print e
			continue

		if args.error:
			#print amp, lae_fibnum, ralae, declae, wave
                        diff1 = random.randint(0, len(ffss)-1)
                        amp, lae_fibnum, ralae, declae = amps[diff1], fibnums[diff1], ras[diff1], decs[diff1]
                        wave = wave #CHANGE THIS!!!! random.uniform(max(def_wave[0], wave-100), min(def_wave[-1], wave+100))
			#print "\n new random amp, fibnum, ra, dec, wave: "
			#print amp, lae_fibnum, ralae, declae, wave, "\n"

		# flag continuum sources
		ffss[ffss==0] = np.nan
		skysubs[skysubs==0] = np.nan
		here = abs(def_wave - wave) <= 100.
		#print "\n#######################################################"
		#before.write(str(np.nanmedian(ffss[:,here]))+ "\t"+str( np.nanmedian(skysubs[:,here]))+"\n")
		#print "#######################################################\n"
	
		#print "\nffss.std() ", np.nanstd(ffss)
		#print "skysubs.std() ", np.nanstd(skysubs), "\n"

		if False:
			meanmean = np.nanmedian(ffss[:,600:900], axis=1)
			
			KAPPA = args.kappa #2.7

			firstflag = np.isfinite(meanmean)

			if False in firstflag:
				print "HERE!!!"
			
			ffss = ffss[firstflag]
			meanmean = meanmean[firstflag]
			ras, decs, thrus, skysubs = ras[firstflag], decs[firstflag], thrus[firstflag], skysubs[firstflag]
			errorframe = errorframe[firstflag]
			fibnums = fibnums[firstflag]
			amps = amps[firstflag]

			biw, biscale = biweight_location(meanmean), biweight_scale(meanmean)
			print "biw, biscale = ", biw, biscale
			if ~(np.isfinite(biw)&np.isfinite(biscale)):
				print meanmean[np.isfinite(meanmean)].size/float(meanmean.size)
				continue
		

			if False:
				plt.figure()
				plt.hist([np.nanmean(ffss, axis=1), meanmean], label=["mean", "median"])
				plt.axvline(biw)
				plt.legend()
				plt.axvline(biw+KAPPA*biscale)
				plt.axvline(biw+2.7*biscale)
				plt.axvline(biw+1.7*biscale)
				plt.show()

			flag = meanmean < biw + KAPPA*biscale

			newflagged = []		
			# flag the fiber before and after a flagged fiber
			if True:
				for i in range(len(flag)):
					if not flag[i]:
						if i in newflagged:
							continue
						else:
							if i%112==0& i!=0 & flag[i+1]:
								flag[i+1] = False
								newflagged.append(i+1)
							elif i%112==111 & flag[i-1]:
								flag[i-1] = False
								newflagged.append(i-1)
							elif i == 0 :
								if flag[i+1]:
									flag[i+1] = False
									newflagged.append(i+1)
							elif i!=flag.size-1:
								if flag[i-1]:
									flag[i-1] = False
									newflagged.append(i-1)
								if flag[i+1]:
									flag[i+1] = False
									newflagged.append(i+1)



		nextfibers = (fibnums >= lae_fibnum - 1)&(fibnums <= lae_fibnum + 1)&(amps == amp)
		
		#print "adjacent fibers : ", fibnums[nextfibers].shape

		flag = ~nextfibers #flag & (~nextfibers) # this was before the shot-wise flagging
		#print ffss.shape, ffss[flag].shape

		ffss = ffss[flag]
		ras, decs, thrus, skysubs = ras[flag], decs[flag], thrus[flag], skysubs[flag]

		#rint "\n mean of all pixels: ", np.nanmedian(ffss), " original ", np.nanmedian(skysubs),  "\n"
		#medfile.write(str(np.nanmedian(ffss))+"	"+str(np.nanmedian(skysubs))+"\n")
		weights = np.ones(errorframe[flag].shape) # errorframe[flag]**(-2)
		#weights[~np.isfinite(weights)] = 0.

	
		radecs = zip(ras, decs)
		ragrid, decgrid = global_ragrid + ralae, global_decgrid + declae
        	#np.arange(ralae-MAXAP/3600., ralae+(MAXAP+1.)/3600., 1./3600), np.arange(declae-MAXAP/3600., declae+(MAXAP+1.)/3600., 1./3600)
		#print radecs
		#ragrid, decgrid = np.arange(np.nanmin(ras)-1./3600, np.nanmax(ras)+1./3600, 1./3600), np.arange(np.nanmin(decs)-1./3600, np.nanmax(decs)+1./3600, 1./3600)

		z_lae = wave/1215.67 - 1
		#print "z lae", z_lae
		distances_lae = abs(cosmo.comoving_distance(z_lae) - distances_here) /u.Mpc * 1000
		
		#print "shape, min, max distances_lae ",  distances_lae.shape, distances_lae.min(), distances_lae.max()

		comoving_kpc_arcmin = cosmo.kpc_comoving_per_arcmin(z_lae)
		comoving_kpc_deg = comoving_kpc_arcmin * 60.

		radiff, decdiff = ras - ralae, decs - declae
		diff = np.sqrt(radiff**2+decdiff**2)
		distance_transverse =  diff * comoving_kpc_deg * u.arcmin/u.kpc		

		#print "min, max diff: ", distance_transverse.min(), distance_transverse.max()

		total_diff = np.zeros((distance_transverse.shape[0], 1036))
		for idx1 in range(total_diff.shape[0]):
			total_diff[idx1] = np.sqrt(distance_transverse[idx1]**2+distances_lae**2)	

		#print total_diff.shape
		#print total_diff
		#print "min, max total diff: ", total_diff.min(), total_diff.max()
		
		radial = []
		for i, ap in enumerate(distances[:-1]):
			here = (ap < distance_transverse)&(distance_transverse <= distances[i+1])
			#print ap, here.size, here[here].size
			weights_here = weights[here][:,(def_wave >= wave - 100.)&(def_wave <= wave -10.)]
			ffss_here = ffss[here][:,(def_wave >= wave - 100.)&(def_wave <= wave - 10.)]
			skysubs_here = skysubs[here][:,(def_wave >= wave - 100.)&(def_wave <= wave - 10.)] # CHANGE BACK TO 5
			weights_here2 = weights[here][:,(def_wave >= wave + 10.)&(def_wave <= wave + 100.)]
			ffss_here2 = ffss[here][:,(def_wave >= wave + 10.)&(def_wave <= wave + 100.)]
			skysubs_here2 = skysubs[here][:,(def_wave >= wave - 100.)&(def_wave <= wave + 100.)] # CHANGE BACK TO 5
			#weights_here = weights[here][:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)]
			#ffss_here = ffss[here][:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)]
			#skysubs_here = skysubs[here][:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)]
				
			for element1, element2, element3 in zip(ffss_here.flatten(), weights_here.flatten(), skysubs_here.flatten()):

				radial_fibers[i].append(element1)
				radial_weights[i].append(element2)
				radial_skysubs[i].append(element3)

			for element1, element2, element3 in zip(ffss_here2.flatten(), weights_here2.flatten(), skysubs_here2.flatten()):

				radial_fibers[i].append(element1)
				radial_weights[i].append(element2)
				radial_skysubs[i].append(element3)
			ffss_here[ffss_here==0] = np.nan
			radial.append(np.nansum(ffss_here*weights_here)/np.nansum(weights_here))
		radial = np.array(radial)
		#radial = radial - np.nansum(ffss[:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)]*weights[:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)])/np.nansum(weights[:,(def_wave >= wave - 5.)&(def_wave <= wave + 5.)])
		#midwave = wave
		#midcoord = ralae, declae #(midra, middec)
		#sigma = 0.0
		#linewidth = 5.
		#fluxmap =  get_fluxmap(midwave, midcoord, radecs, ffss, weights, sigma, linewidth=linewidth)
		#print "min and max fluxmap junior ", np.nanmin(fluxmap), np.nanmax(fluxmap)

		fluxmaps.append(radial)
		print "yeah!\n"
	except Exception as E:
		print E
		continue
#medfile.close()
#print "written to medfile"
#before.close()
#print "written to before"

radial_fibers = np.array(radial_fibers)
radial_weights = np.array(radial_weights)
radial_skysubs = np.array(radial_skysubs)

print "radial fibers and weights shape: \n"
print radial_fibers.shape, radial_weights.shape, radial_skysubs.shape, "\n"

total = [np.nansum(np.array(radial_fibers[i])*np.array(radial_weights[i]))/np.nansum(radial_weights[i]) for  i in range(radial_fibers.shape[0])]
total_orig = [np.nansum(np.array(radial_skysubs[i])*np.array(radial_weights[i]))/np.nansum(radial_weights[i]) for  i in range(radial_fibers.shape[0])]

total_std = np.array([np.sqrt(np.nanstd(radial_fibers[i])**2*np.nansum(np.array(radial_weights[i])**2)/(np.nansum(radial_weights[i]))**2 )
				for i in range(radial_fibers.shape[0])])
print total_std
total_orig_std = np.array([np.sqrt(np.nanstd(radial_skysubs[i])**2*np.nansum(np.array(radial_weights[i])**2)/(np.nansum(radial_weights[i]))**2 )
				for i in range(radial_fibers.shape[0])])

total = np.array(total)

print total.shape
print "total", total

if args.error:
	ascii.write([distances[1:], total, total_orig, total_std, total_orig_std],
	 "3d-radialerror-dustin-{}.dat".format(diff1), names=["distances [ckpc]","counts", "counts(original)", "std", "std(original)"], overwrite=True)
elif False:
	ascii.write([distances[1:], total, total_orig, total_std, total_orig_std],
	 "3d-radialsum-dustin.dat", names=["distances [ckpc]","counts", "counts(original)", "std", "std(original)"], overwrite=True)
else:
	ascii.write([distances[1:], total, total_orig, total_std, total_orig_std],
	 "3d-radialsum-dustin-100range.dat", names=["distances [ckpc]","counts", "counts(original)", "std", "std(original)"], overwrite=True)

#fluxmaps = np.nanmedian(fluxmaps, axis=0)
#print fluxmaps.shape


"""xc, yc = np.floor(fluxmap.shape[1]/2.), np.floor(fluxmap.shape[0]/2.)
print xc, yc
r = 10.

xx = np.arange(fluxmap.shape[1]) - xc
yy = np.arange(fluxmap.shape[0]) - yc
XX,YY = np.meshgrid(xx,yy)
dd = np.sqrt(XX**2. + YY**2.)

radial = []
for r in np.arange(1, 30, 1):

	ii = (dd >= r-1)&(dd <= r)
	medianflux = np.nanmedian(fluxmap[ii])
	print "median flux in aperture "+str(r)+" : "+str(medianflux)
	radial.append(medianflux)
	print np.unique(ii)

radial = np.array(radial)
"""
#ascii.write([apertures[1:], fluxmaps], "radial-dustin-20189-3f-lowkappa-moreshots.dat", names=["apertures [deg]","counts"])

if False:
	plt.figure(figsize=(15,5))
	plt.subplot(211)
	plt.ylabel("counts")
	plt.xlabel("aperture [arcsec]")
	plt.plot(apertures[1:]*3600., fluxmaps, drawstyle='steps-mid')
	plt.axhline(0, linestyle="--", color="grey")
	plt.subplot(212)
	plt.plot(apertures[1:]*3600., fluxmaps, drawstyle='steps-mid')
	plt.ylabel("counts")
	plt.xlabel("aperture [arcsec]")
	plt.yscale("log")
	plt.savefig("radial-dustin-20189-3f-lowkappa.png", bbox_inches="tight")

PLOT = False
if PLOT:
	plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	plt.imshow(gaussian_filter(fluxmap[::-1,::-1], sigma=1), vmin=0., vmax=2, cmap="inferno", aspect="auto", interpolation="none", zorder=1)
	#plt.contour(gaussian_filter(fluxmap[::-1,::-1], sigma=1), levels=(range(5)), colors=["white"], alpha=0.4, color="white")
	#plt.plot([np.where(abs(ragrid-x[0])<=1./3600)[0][0] for x in laes], [np.where(abs(decgrid[::-1]-x[1])<=1./3600)[0][0] for x in laes], "*", color="cyan", zorder=2)
	plt.title(r"Cosmos C {}, {} $\pm$ {} $\AA$".format(ifu, midwave, linewidth))
	plt.xlabel("RA [deg]")
	plt.ylabel("Dec [deg]")
	plt.xticks(range(len(ragrid))[::10],[round(x,3) for x in ragrid[::-1][::10]])
	plt.yticks(range(len(decgrid))[::10], [round(x,3) for x in decgrid[::-1][::10]])
	plt.colorbar()
	#plt.contour(gaussian_filter(fluxmap[::-1,::-1], sigma=1), levels=(range(5)), colors=["white"], alpha=0.3)
	plt.subplot(1,2,2)
	plt.imshow(gaussian_filter(-1.*fluxmap[::-1,::-1], sigma=1), vmin=0., vmax=2, cmap="inferno", aspect="auto", interpolation="none", zorder =1)
	#plt.plot([np.where(abs(ragrid-x[0])<=1./3600)[0][0] for x in radecs[::3]], [np.where(abs(decgrid[::-1]-x[1])<=1./3600)[0][0] for x in radecs[::3]], "o", zorder = 2)
	#plt.plot([np.where(abs(ragrid-x[0])<=1./3600)[0][0] for x in blobshere[::1]], [np.where(abs(decgrid[::-1]-x[1])<=1./3600)[0][0] for x in blobshere[::1]], "o", zorder = 3, alpha = 0.3, color="white")
	plt.title(r"negative values, {} $\pm$ {} $\AA$".format(midwave, linewidth))
	plt.xlabel("RA [deg]")
	plt.ylabel("Dec [deg]")
	plt.colorbar()
	plt.contour(gaussian_filter(fluxmap[::-1,::-1], sigma=1), levels=(range(5)), colors=["white"], alpha=0.4, color="white")
	plt.xticks(range(60)[::10],[round(x,3) for x in ragrid[::-1][::10]])
	plt.yticks(range(60)[::10], [round(x,3) for x in decgrid[::-1][::10]])
	#plt.colorbar()
	plt.xlabel("RA [deg]")
	plt.ylabel("Dec [deg]")
	plt.savefig("try-stack-dustin-new.png", bbox_inches="tight")
