# get PSF - running biweight of normalized radial star profiles

from astropy.io import ascii

from astropy.stats import biweight_location, biweight_scale
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from matplotlib import cm

import glob

# update matplotlib params for bigger font
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral'}

pylab.rcParams.update(params)
plt.style.use("seaborn-whitegrid")

from hetdex_api.shot import *
def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table

def distsq(midra, middec, ras, decs):

    """returns the angular distances in arcseconds"""

    deccos = np.cos(middec*np.pi/180.)
    return (((ras-midra)*deccos)**2 + (decs-middec)**2)*(3600)**2

def moffat_arcsec(rsq, amp, index, fwhm):
    """returns the moffat function with alpha=index and fwhm in arcseconds."""
    gamma = fwhm/(2 * np.sqrt(2**(1/index)-1))
    return amp * (1 + rsq / gamma**2)**(-index)

def moffat_arcsec_3(rsq, amp, fwhm):
    """returns the moffat function with alpha=index and fwhm in arcseconds."""
    index = 3.0
    gamma = fwhm/(2 * np.sqrt(2**(1/index)-1))
    return amp * (1 + rsq / gamma**2)**(-index)

def moffat_grid(xgrid, ygrid, amp, fwhm):
    return moffat_arcsec_3(xgrid**2+ygrid**2, amp, fwhm)

def integrated_fibers(rsq, amp, fwhm):
    moffgrid = moffat_grid(xgrid, ygrid, amp, fwhm)
    summed = []
    for r in np.sqrt(rsq/2.0):
        rhere = (xgrid-r)**2+(ygrid-r)**2<1.0
        summed.append(np.nanmean(moffgrid[rhere]))
    return summed

# load star files
startabs = glob.glob("/data/05865/maja_n/Jupyter/ff2.1_stars/*.tab")
laetabs = glob.glob("/data/05865/maja_n/Jupyter/laetabs/*.tab")
laeinfo = ascii.read("/data/05865/maja_n/Jupyter/catalogs/goodlaes.cat")

def fit_psf(r, amplitude, fwhm):
    ys = np.interp(r/fwhm, psf_shape["r/fwhm"], psf_shape["runbiw_filt"])
    return amplitude * ys

PSFSHAPE = False # switch to fit the PSF shape instead of moffat
if PSFSHAPE:
    # get PSF shape
    psf_shape = ascii.read("/data/05865/maja_n/Jupyter/PSF_runbiw.dat")

INTEGRATE = False  # switch to integrate over fiber areas
if INTEGRATE:
    xgrid = np.array([np.arange(-1, 10, 0.1) for i in range(110)])
    ygrid = xgrid.T

STARS = True
if STARS:
    tabs = startabs
    fluxkey = "flux"
    errkey = "std"
else:
    tabs = laetabs
    fluxkey = "full_skysub"
    errkey = "std/sqrt(N-q)"

results = []
pcovs = []
shotresults = []
shotidx = -1
oldshot = "111111"
testdists = np.arange(0., 13., 0.2)
starids = []
normalized = []
distances = []
for sf in tabs[:]:
    try:
        if STARS:
            shot = sf.split("_")[2]
            starid = sf.split("_")[3][:-4]
        else:
            detectid = int(sf.split("/")[-1][:-4])


        ############################ Check if this is actually ok data #############################

        a = ascii.read(sf)
        a = a[np.isfinite(a[fluxkey])&(a[fluxkey]!=0.0)] #a[np.isfinite(a["flux"])&(a["flux"]!=0.0)]
        if len(a)<1:
            continue
        if STARS:
            rsq = distsq(a[0]["star_ra"], a[0]["star_dec"], a["ra"].data, a["dec"].data) # in arcseconds
        else:
            midra, middec = laeinfo["ra"][laeinfo["detectid"]==detectid], laeinfo["dec"][laeinfo["detectid"]==detectid]
            rsq = distsq(midra, middec, a["ra"], a["dec"])
        a = a[np.sqrt(rsq) < 13.]
        rsq = rsq[np.sqrt(rsq) < 13.]
        if len(rsq)<1:
            continue
        elif np.nanmin(rsq)>1.5**2:
            continue

        ########################### fit a function to the radial profiles ###################

        if INTEGRATE:
            p0 = [np.nanmax(a[fluxkey].data), 1.7]
            popt, pcov = curve_fit(integrated_fibers, rsq, a[fluxkey], sigma=a[errkey], p0=p0)

        elif PSFSHAPE:
            p0 = [np.nanmax(a[fluxkey]), 1.7]
            popt, pcov = curve_fit(fit_psf, np.sqrt(rsq), a["flux"], sigma=a["std"], p0=p0)

        else: # fit a moffat function with beta = 3
            p0 = [np.nanmax(a[fluxkey].data), 1.7]
            popt, pcov = curve_fit(moffat_arcsec_3, rsq, a[fluxkey], sigma=a[errkey], p0=p0)

        results.append(popt)
        pcovs.append(pcov)
        if MOCK:
            normalized.append(moff/popt[0])
        else:
            normalized.append(a[fluxkey].data/popt[0])
        distances.append(rsq)
        shotresults.append(shotidx)#int(shot))#idx)
        if STARS:
            starids.append(starid)
        else:
            starids.append(detectid)
    except RuntimeError as e:
        print(e)
        pass

normalized = np.array(normalized)
distances = np.array(distances)
amps, fwhms = np.array([x[0] for x in results]), np.array([x[1] for x in results])
shots = shotresults

amps, fwhms = np.array([x[0] for x in results]), np.array([x[1] for x in results])
amperr, fwhmerr = np.array([np.sqrt(x[0][0]) for x in pcovs]), np.array([np.sqrt(x[1][1]) for x in pcovs])

withinbouds = (fwhms<4)&(fwhms>1.2)&(amps>1)&(amps<300)&(fwhms!=1.7)
print("kept {} out of {} stars.".format(len(fwhms[withinbouds]), len(fwhms))

# normalize radial profiles and r -> r/FWHM
roverfwhm = []
norm_values = []
for x, y, fwhm in zip(distances[withinbouds], normalized[withinbouds], fwhms[withinbouds]):
    roverfwhm.append(np.sqrt(x)/fwhm)
    norm_values.append(y)

roverfwhm = np.concatenate(roverfwhm)
norm_values = np.concatenate(norm_values)
order = np.argsort(roverfwhm)
roverfwhm = roverfwhm[order]
norm_values = norm_values[order]

# get the PSF
medfilt = []
lower = []
upper = []
Ns = []
errs = []
xbins = np.arange(0, 5, 0.02)
for x, y in zip(xbins[:-1], xbins[1:]):
    here = (roverfwhm>=x)&(roverfwhm<y)
    loc = biweight_location(norm_values[here])
    scale = biweight_scale(norm_values[here])
    N = len(here[here])
    Ns.append(N)
    medfilt.append(loc)
    err = np.sqrt(scale / (N-1))
    lower.append(loc-err)
    upper.append(loc+err)
    errs.append(err)


SAVE = False

# plot the result
plt.figure()
plt.plot(roverfwhm, norm_values, ".k", alpha=0.02)
plt.axhline(0, color="gray", linestyle=":")
plt.plot(xbins[:-1]+0.005, moffat_arcsec_3((xbins[:-1]+0.005)**2, 1, 1), color="r", label="Moffat function")
plt.plot(xbins[:-1]+0.005, medfilt, color="blue", label="PSF")#running biweight")
plt.fill_between(xbins[:-1]+0.005, lower, upper, alpha=0.4)
plt.ylim(-0.1, 1.1)
plt.xlim(0, 3)
plt.ylabel("normalized flux")
plt.xlabel("r / FWHM")
plt.legend(bbox_to_anchor=(1, 0.8), frameon=True)
if SAVE:
    plt.savefig("plots/running_biweight_normed_psf_ff2.1_thesis.png", bbox_inches="tight")
else:
    plt.show()

if SAVE:
    new_psf = ascii.read("PSF_runbiw.dat")
    new_psf["runbiw_fullframe_2.1"] = medfilt
    new_psf["runbiw_fullframe_err_2.1"] = errs
    ascii.write(new_psf, "PSF_runbiw.dat")
