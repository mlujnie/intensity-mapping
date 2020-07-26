# import packages
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

import glob

import tables as tb

import numpy as np

def_wave = np.arange(3470., 5542., 2.)

def get_badamps(shot):
	bads = ascii.read("/work/05865/maja_n/wrangler/im2d/badifus.dat")
	bads = bads["ifuamp"][bads["shot"]==shot]
	badlist = list(bads)
	for bad in bads:
		if bad[-2:] =="AA":
			badlist.append(bad[:-2]+"LL")
			badlist.append(bad[:-2]+"LU")
			badlist.append(bad[:-2]+"RL")
			badlist.append(bad[:-2]+"RU")
	return np.array(badlist)

def load_shot(shot):
	filename = '/work/03946/hetdex/hdr1/reduction/data/{}.h5'.format(shot)
	fileh = tb.open_file(filename, "r")
	table = Table(fileh.root.Data.Fibers.read())
	return table

def get_flexnum(shot, opt="normal"):
	ff = glob.glob("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_???_???_??.fits".format(shot))
	multis, fibers, exps = [], [], []
	bads = get_badamps(shot)
	for counter, fin in enumerate(ff):
		multi = fin.split("/")[-1][:-5]
		exps.append(fin.split("/")[-2])
		multis.append(multi)
		if multi[10:13]+multi[18:20] in bads:
			fibers.append(np.zeros((112, 1036)))
		else: 
			fibers.append(fits.getdata(fin))
		
	exps = np.array(exps)
	multis = np.array(multis)
	fibers = np.array(fibers)
	fibers[fibers==0] = np.nan

	fibers = np.concatenate(fibers)


	# getting the number of continuum fibers to exclude

	if opt=="normal":
		filename = 'flexnum-biloc.dat'
		wavehere = def_wave > 0.
		meanmean = np.nanmedian(fibers, axis=1)
		sorted_mean = np.argsort(meanmean)
		greatersix = np.where(meanmean[sorted_mean] > -6.)[0][0]
		finites = meanmean[np.isfinite(meanmean)].size

	elif opt=="lower":
		filename = 'flexnum-biloc-lower.dat'
		wavehere = (def_wave > 3800)&(def_wave <= 4500)
		meanmean = np.nanmedian(fibers[:,wavehere], axis=1)
		sorted_mean = np.argsort(meanmean)
		greatersix = np.where(meanmean[sorted_mean] > -6.)[0][0]
		finites = meanmean[np.isfinite(meanmean)].size 

	elif opt=="upper":
		filename = 'flexnum-biloc-upper.dat'
		wavehere = (def_wave > 4500)&(def_wave <= 5200)
		meanmean = np.nanmedian(fibers[:,wavehere], axis=1)
		sorted_mean = np.argsort(meanmean)
		greatersix = np.where(meanmean[sorted_mean] > -6.)[0][0]
		finites = meanmean[np.isfinite(meanmean)].size 

	flexes = ascii.read("/work/05865/maja_n/wrangler/im2d/"+filename)

	if shot in flexes['shot']:
		print('found shot in '+filename)
		flex = flexes['flexnum'][flexes['shot']==shot]
		if not type(flex)==int:
			flex = flex[0]


		#wlhere = (def_wave > 4000)&(def_wave < 5000)
		#redmed = np.nanmedian(fibers[:,wlhere], axis=1)
		#fibers[abs(redmed)>=10.] = np.nan
		#maxim = meanmean[sorted_mean[flex]]
		#ftmp = fibers[sorted_mean[:flex]]	
		#medmed = np.nanmedian(ftmp)
		#biloc = biweight_location(ftmp[np.isfinite(ftmp)])
		#tmp = open('biloc.txt','a')
		#tmp.write(str(shot)+'	'+str(medmed)+"	"+str(biloc)+'\n')
		#tmp.close()
	else:
		written = False
		print('computing flex.')
		# include flexible parameter, such that median(medians) = 0
		#fibers_con = np.concatenate(fibers)

		for flex in range(int(finites*0.85), int(finites), 15):
		#for maxim in np.arange(2, 7, 0.1)[::-1]:
			#ftmp = fibers[abs(meanmean)<maxim]
			ftmp = fibers[sorted_mean[greatersix:flex]][:,wavehere] #np.nanmedian(np.nanmedian(fibers_con[sorted_mean[:flex]], axis=0))
			medmed = biweight_location(ftmp[np.isfinite(ftmp)])
			if medmed >= 0:
				maxim = meanmean[sorted_mean[flex]]
				#if ~np.isfinite(meanmean[sorted_mean][-1]):
				#	flex = flex + np.size(meanmean[~np.isfinite(meanmean)])
				tmp = open(filename,'a')
				tmp.write(str(shot)+'	'+str(flex)+'	'+str(maxim)+'	'+str(medmed)+'	'+str(flex/float(finites))+'\n')
				#tmp.write(str(shot)+'	'+str(maxim)+'	'+str(medmed)+'	'+str(float(meanmean[abs(meanmean)<maxim].size)/finites)+'\n')
				tmp.close()
				written = True
				#print flex, medmed
				print('percentage: ', float(flex)/finites)
				break
		if not written:
			#if ~np.isfinite(meanmean[sorted_mean][-1]):
			#	flex = flex + np.size(meanmean[~np.isfinite(meanmean)])
			tmp = open(filename,'a')
			tmp.write(str(shot)+'   '+str(flex)+'	'+str(medmed)+'\n')
			tmp.close()
	


def get_ffss_noflag(shot):
	table = load_shot(shot)

	xrt = get_xrt_time(shot)
	
	ff = glob.glob("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_???_???_??.fits".format(shot))
	if len(ff)==0:
		print("nothing found in {}".format("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_???_???_??.fits".format(shot)))
	multis, fibers, exps = [], [], []
	bads = get_badamps(shot)
	for counter, fin in enumerate(ff):
		multi = fin.split("/")[-1][:-5]
		exps.append(fin.split("/")[-2])
		multis.append(multi)
		if multi[10:13]+multi[18:20] in bads:
			fibers.append(np.zeros((112, 1036)))
		else: 
			fibers.append(fits.getdata(fin))
		
	exps = np.array(exps)
	multis = np.array(multis)
	fibers = np.array(fibers)
	fibers[fibers==0] = np.nan

	# getting the number of continuum fibers to exclude
	meanmean = np.nanmedian(np.concatenate(fibers), axis=1)
	sorted_mean = np.argsort(meanmean)
	finites = meanmean[np.isfinite(meanmean)].size


	flexes = ascii.read('/work/05865/maja_n/wrangler/im2d/flexnum.dat')
	if shot in flexes['shot']:
		print('found shot in flexnum.dat')
		flex = flexes['flexnum'][flexes['shot']==shot]
		if not type(flex)==int:
			flex = flex[0]
	else:
		print('Error: did not find flexnum.')
		return 0,0,0,0,0,0,0,0,0
		print('computing flex.')
		# include flexible parameter, such that median(medians) = 0
		fibers_con = np.concatenate(fibers)
		for flex in range(int(finites*0.85), int(finites), 25):
			medmed = np.nanmedian(np.nanmedian(fibers_con[sorted_mean[:flex]], axis=0))
			if medmed >= 0:
				tmp = open('flexnum.dat','a')
				tmp.write(str(shot)+'   '+str(flex)+'\n')
				tmp.close()
				#print flex, medmed
				print('percentage: ', float(flex)/finites)
				break
	
	maxim = meanmean[sorted_mean[flex]]
	#medmed = np.nanmedian(np.nanmedian(np.concatenate(fibers)[sorted_mean[:flex]], axis=0))
	#tmp = open("maxim.dat","a")
	#tmp.write(str(shot)+"	"+str(maxim)+"	"+str(medmed)+"\n")
	#tmp.close()

	#print("\n{}\n".format(maxim))
	flag = meanmean <=  10000. #maxim

	flag = np.array(np.split(flag, len(ff)))

	amps = np.array([x[18:20] for x in multis])
	ifus = np.array([x[10:13] for x in multis])

	multiname, ffss, ras, decs = [], [], [], []
	throughputs, skysubs, errors, fibnums = [], [], [], []
	fluxcal = []

	for i in range(len(multis)):
		amp = amps[i]
		ifu = ifus[i]
		idx = (table["multiframe"]==multis[i])&(table["expnum"]==int(exps[i][-1]))
		if len(idx[idx])==0:
			continue
		
		ifux, ifuy = table['ifux'][idx], table['ifuy'][idx]
		noedge = (abs(ifux) <= 23) # change this !!!
		flag[i] *= noedge
		noedge = (abs(ifuy) <= 23) # change this !!!
		flag[i] *= noedge

		fibers[i][~flag[i]] = 0.

		#print(f'fibers {i} == 0 : {fibers[i][fibers[i]==0].size/fibers[i].size}')
		f2f = np.array([np.interp(def_wave, x, y) for x, y in zip(table["wavelength"][idx], table["fiber_to_fiber"][idx])])
		throughput = table["Throughput"][idx]

		A2A = table["Amp2Amp"][idx]
		a2a = np.ones(A2A.shape)
		a2a[A2A <= 0] = 0

		ra, dec = table["ra"][idx], table["dec"][idx]

		skysub = []
		for x, y in zip(table["wavelength"][idx], table["sky_subtracted"][idx]):
			try:
				skysub.append(np.interp(def_wave, x[y!=0], y[y!=0], left=0.0, right=0.0))
			except ValueError:
				skysub.append(np.zeros(def_wave.shape))
		skysub = np.array(skysub)
		skysub[~flag[i]] = 0.0
		skysubs.append(skysub / (f2f * xrt[(ifu, amp)](def_wave)) * a2a)
		
		gain = table["calfib"][idx] / skysub 
		gain[~np.isfinite(gain)] = 0.
		
		biscale = biweight_scale(gain[np.isfinite(gain)])
		biloc = biweight_location(gain[np.isfinite(gain)])
		these = abs(gain-biloc) > 6*biscale
		gain[these] = 0.
		fluxcal.append(fibers[i]*gain /  (f2f * xrt[(ifu, amp)](def_wave)) * a2a)


		error = np.array([np.interp(def_wave, x, y, left=0.0, right=0.0) for x, y in zip(table["wavelength"][idx], table["error1Dfib"][idx])])
		errors.append(error)

		fibnums.append(table["fibidx"][idx])
		
		ffss.append( fibers[i] / (f2f * xrt[(ifu, amp)](def_wave)) * a2a)
		ras.append(ra)
		decs.append(dec)
		throughputs.append(throughput)

	amps = [[x for i in range(112)] for x in amps]
	ffss = np.array(ffss)
	return np.array(ffss), np.array(ras), np.array(decs), np.array(throughputs), np.array(skysubs), np.array(errors), np.array(fibnums), np.array(amps), np.array(multis), np.array(fluxcal)


def get_ffss(shot, asymcut=True):
	table = load_shot(shot)

	xrt = get_xrt_time(shot)
	
	ff = glob.glob("/work/05865/maja_n/stampede2/ffskysub/{}/exp0?/multi_???_???_???_??.fits".format(shot))
	multis, fibers, exps = [], [], []
	bads = get_badamps(shot)
	for counter, fin in enumerate(ff):
		multi = fin.split("/")[-1][:-5]
		exps.append(fin.split("/")[-2])
		multis.append(multi)
		if multi[10:13]+multi[18:20] in bads:
			fibers.append(np.zeros((112, 1036)))
		else: 
			fibers.append(fits.getdata(fin))
		
	exps = np.array(exps)
	multis = np.array(multis)
	fibers = np.array(fibers)
	fibers[fibers==0] = np.nan

	# getting the number of continuum fibers to exclude
	#meanmean = np.nanmedian(np.concatenate(fibers), axis=1)
	#sorted_mean = np.argsort(meanmean)
	#finites = meanmean[np.isfinite(meanmean)].size

	wlhere_l = (def_wave > 3800.)&(def_wave <= 4500.)
	wlhere_u = (def_wave > 4500.)&(def_wave <= 5200.)

	meanmean_l = np.nanmedian(np.concatenate(fibers)[:,wlhere_l], axis=1)
	meanmean_u = np.nanmedian(np.concatenate(fibers)[:,wlhere_u], axis=1)
	
	#flexes = ascii.read('flexnum-biloc.dat')
	flexes_l = ascii.read('/work/05865/maja_n/wrangler/im2d/flexnum-biloc-lower.dat')
	flexes_u = ascii.read('/work/05865/maja_n/wrangler/im2d/flexnum-biloc-upper.dat')
	if shot in flexes_l['shot']:
		print('found shot in flexnum-biloc-lower.dat')
		#flex = flexes['flexnum'][flexes['shot']==shot]
		#if not type(flex)==int:
		#	flex = flex[0]
		maxim_l = flexes_l["maxim"][flexes_l["shot"]==shot]
		if not type(maxim_l)==int:
			maxim_l = maxim_l[0]
	if shot in flexes_u['shot']:
		print('found shot in flexnum-biloc-upper.dat')
		#flex = flexes['flexnum'][flexes['shot']==shot]
		#if not type(flex)==int:
		#	flex = flex[0]
		maxim_u = flexes_u["maxim"][flexes_u["shot"]==shot]
		if not type(maxim_u)==int:
			maxim_u = maxim_u[0]
	else:
		print('Error: did not find flexnum.')
		if False: #### CHANGE THIS!!!
			return 0,0,0,0,0,0,0,0,0
			print('computing flex.')
			# include flexible parameter, such that median(medians) = 0
			fibers_con = np.concatenate(fibers)
			for flex in range(int(finites*0.85), int(finites), 25):
				medmed = np.nanmedian(np.nanmedian(fibers_con[sorted_mean[:flex]], axis=0))
				if medmed >= 0:
					tmp = open('flexnum.dat','a')
					tmp.write(str(shot)+'   '+str(flex)+'\n')
					tmp.close()
					#print flex, medmed
					print('percentage: ', float(flex)/finites)
					break
	
	#maxim = meanmean[sorted_mean[flex]]
	#ftmp = np.concatenate(fibers)[abs(meanmean)<=maxim]
	#ftmp2 = np.concatenate(fibers)[(meanmean>-6.)&(meanmean<maxim)]
	#medmed = biweight_location(ftmp[np.isfinite(ftmp)])
	#medmed2 = biweight_location(ftmp2[np.isfinite(ftmp2)])
	#print("medmed: ", medmed)
	#print("medmed2: ", medmed2)
	#print("\n")
	
	#tmp = open("maxim.dat","a")
	#tmp.write(str(shot)+"	"+str(maxim)+"	"+str(medmed)+"\n")
	#tmp.close()

	#print("\n{}\n".format(maxim))

	if not asymcut:
		print("cutting symmetrically from -5 to 5.")
		maxim_l, maxim_u = 5., 5.
		min_l, min_u = -1*maxim_l, -1*maxim_u
	else:
		min_l, min_u = -6., -6.

	flag1 = (meanmean_l <=  maxim_l) & (meanmean_l > min_l)
	flag2 = (meanmean_u <= maxim_u) & (meanmean_u > min_u)
	flag = flag1 * flag2

	ftmp = np.concatenate(fibers)[flag]
	print("\nbiloc remaining: {}\n".format(biweight_location(ftmp[np.isfinite(ftmp)])))

	flag = np.array(np.split(flag, len(ff)))

	amps = np.array([x[18:20] for x in multis])
	ifus = np.array([x[10:13] for x in multis])

	multiname, ffss, ras, decs = [], [], [], []
	throughputs, skysubs, errors, fibnums = [], [], [], []
	fluxcal = []

	for i in range(len(multis)):
		amp = amps[i]
		ifu = ifus[i]
		idx = (table["multiframe"]==multis[i])&(table["expnum"]==int(exps[i][-1]))
		if len(idx[idx])==0:
			continue
		
		ifux, ifuy = table['ifux'][idx], table['ifuy'][idx]
		noedge = (abs(ifux) <= 22) # change this !!! to 23
		flag[i] *= noedge
		noedge = (abs(ifuy) <= 22) # change this !!! to 23
		flag[i] *= noedge

		fibers[i][~flag[i]] = 0.

		#print(f'fibers {i} == 0 : {fibers[i][fibers[i]==0].size/fibers[i].size}')
		f2f = np.array([np.interp(def_wave, x, y) for x, y in zip(table["wavelength"][idx], table["fiber_to_fiber"][idx])])
		throughput = table["Throughput"][idx]

		A2A = table["Amp2Amp"][idx]
		a2a = np.ones(A2A.shape)
		a2a[A2A <= 0] = 0

		ra, dec = table["ra"][idx], table["dec"][idx]

		skysub = []
		for x, y in zip(table["wavelength"][idx], table["sky_subtracted"][idx]):
			try:
				skysub.append(np.interp(def_wave, x[y!=0], y[y!=0], left=0.0, right=0.0))
			except ValueError:
				skysub.append(np.zeros(def_wave.shape))
		skysub = np.array(skysub)
		skysub[~flag[i]] = 0.
		
		gain = table["calfib"][idx] / skysub 
		gain[~np.isfinite(gain)] = 0.
		
		biscale = biweight_scale(gain[np.isfinite(gain)])
		biloc = biweight_location(gain[np.isfinite(gain)])
		these = abs(gain-biloc) > 6*biscale
		gain[these] = 0.
		fluxcal.append(fibers[i]*gain /  (f2f * xrt[(ifu, amp)](def_wave)) * a2a)

		skysub[~flag[i]] = 0.0
		skysubs.append(skysub / (f2f * xrt[(ifu, amp)](def_wave)) * a2a)
	
		error = np.array([np.interp(def_wave, x, y, left=0.0, right=0.0) for x, y in zip(table["wavelength"][idx], table["error1Dfib"][idx])])
		errors.append(error)

		fibnums.append(table["fibidx"][idx])
		
		ffss.append( fibers[i] / (f2f * xrt[(ifu, amp)](def_wave)) * a2a)
		ras.append(ra)
		decs.append(dec)
		throughputs.append(throughput)

	amps = [[x for i in range(112)] for x in amps]
	ffss = np.array(ffss)
	return np.array(ffss), np.array(ras), np.array(decs), np.array(throughputs), np.array(skysubs), np.array(errors), np.array(fibnums), np.array(amps), np.array(multis), np.array(fluxcal)

def get_closest_date(inpath, shot):
	pp = np.sort(glob.glob(inpath))
	date = int(shot[:-4])
	dates_inpath = np.array([int(x.split("/")[-1]) for x in pp])
	date_diff = date - dates_inpath
	out_date_idx = np.where(date_diff>=0)[0][-1]
	out_date_path = pp[out_date_idx]
	return out_date_path

def get_xrt_time(shot):
	inpath = "/work/05865/maja_n/stampede2/midratio/*"
	outpath = get_closest_date(inpath, shot)
	pattern = outpath+"/{}.dat"
	xrt = {}
	wave = def_wave
	line = 3910
	here1 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 4359
	here2 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 5461
	here3 = np.where((wave>line-10)&(wave<line+10))[0]

	multinames = np.unique([x.split("/")[-1][:-4] for x in glob.glob(pattern.format("*"))])
	SIGMA = 4.
	if True:
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
					#print(xrt_1[~np.isfinite(xrt_1)]
					xrt[key] = interp1d(wave, gaussian_filter(xrt_1, sigma=SIGMA/2.), fill_value=(xrt_1[0],xrt_1[-1]),bounds_error=False)
	else:
			for key in xrt_0.keys():
					xrt[key] = interp1d(wave, xrt_0[key], fill_value=(xrt_0[key][0],xrt_0[key][-1]),bounds_error=False)
	return xrt
 
