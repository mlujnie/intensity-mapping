"""I used this to get the 'integrated PSF function'.
   We fit the 'integrated PSF function'
   to the star fluxes, divide these by the best-fit amplitude, and
   compute a biweight in small bins of r/FWHM."""

import numpy as np
import glob
from astropy.io import ascii
from astropy.stats import biweight_location, biweight_scale
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

import multiprocessing
import time

# whether to save the new PSF file
SAVE = True

def distsq(midra, middec, ras, decs):

    """returns the angular distances in arcseconds"""

    deccos = np.cos(middec*np.pi/180.)
    return (((ras-midra)*deccos)**2 + (decs-middec)**2)*(3600)**2

def integrate_psf(dist, fwhm):

    """integrates the psf function over the fiber area,
    not used at the moment."""

    dist_xy = dist/np.sqrt(2)
    gridrange = np.arange(dist_xy-0.75, dist_xy+0.76, 0.01) # diameter of a fiber is 1.5'' -> radius = 0.75''
    xgrid = np.array([gridrange for i in range(len(gridrange))])
    ygrid = xgrid.T

    fiber_r = np.sqrt((xgrid-dist_xy)**2 + (ygrid-dist_xy)**2)
    disthere = fiber_r <= 0.75

    grid_r = np.sqrt(xgrid**2 + ygrid**2)
    grid_r[~disthere] = np.nan

    psf_grid = psf_func(grid_r/fwhm)
    mean_psf = np.nanmean(psf_grid[disthere])
    return mean_psf

def moffat_arcsec_3(rsq, amp, fwhm):
    """returns the moffat function with alpha=index and fwhm in arcseconds."""
    index = 3.0
    gamma = fwhm/(2 * np.sqrt(2**(1/index)-1))
    return amp * (1 + rsq / gamma**2)**(-index)

def integrate_moffat(distsq, amp, fwhm):

    """integrates the moffat function over the fiber area"""

    dist_xy = np.sqrt(distsq/2)
    gridrange = np.arange(dist_xy-0.75, dist_xy+0.76, 0.01) # diameter of a fiber is 1.5'' -> radius = 0.75''
    xgrid = np.array([gridrange for i in range(len(gridrange))])
    ygrid = xgrid.T

    fiber_r = np.sqrt((xgrid-dist_xy)**2 + (ygrid-dist_xy)**2)
    disthere = fiber_r <= 0.75

    grid_rsq = xgrid**2 + ygrid**2
    grid_rsq[~disthere] = np.nan

    psf_grid = moffat_arcsec_3(grid_rsq, amp, fwhm)
    mean_psf = np.nanmean(psf_grid[disthere])
    return mean_psf

def int_moff(distsq, amp, fwhm):
    """returns integrate_moffat() for an array of distances"""
    return [integrate_moffat(x, amp, fwhm) for x in distsq]

def fit_psf(dist, amp, fwhm):
    return psf_func(dist/fwhm) * amp

# open the current PSF file and interpolate
psf_shape = ascii.read("PSF_runbiw.dat")

# normalize such that it goes to one at r=0
psf_gaus_filt = gaussian_filter(psf_shape["runbiw_int_ff_2.1"], 2)
psf_gaus_filt /= psf_gaus_filt[0]

# normalize so that at r=0.5*FWHM, the PSF = 0.5.
psf_back_func = interp1d(psf_gaus_filt, psf_shape["r/fwhm"])
psf_shape["r/fwhm"] = psf_shape["r/fwhm"]/psf_back_func(0.5)*0.5

# now interpolate
psf_func = interp1d(psf_shape["r/fwhm"], psf_gaus_filt, kind = "cubic")

# find the star flux data
startabs = glob.glob("/data/05865/maja_n/Jupyter/ff2.1_stars/*.tab")

STARS = True
if STARS:
    tabs = startabs
    fluxkey = "flux"
    errkey = "std"
else:
    tabs = laetabs
    fluxkey = "full_skysub"
    errkey = "std/sqrt(N-q)"

# check which stars have good data
star_rsqs = []
star_fluxes = []
star_errs = []
for st in tabs:
    star = ascii.read(st)
    ras, decs = star["ra"], star["dec"]
    star_ra, star_dec = star[0]["star_ra"], star[0]["star_dec"]
    rsqares = distsq(star_ra, star_dec, ras, decs)

    star = star[np.isfinite(star[fluxkey])&(star[fluxkey]!=0.0)]
    if len(star)<1:
        continue
    rsq = distsq(star[0]["star_ra"], star[0]["star_dec"], star["ra"].data, star["dec"].data) # in arcseconds
    max_r = 5.
    star = star[np.sqrt(rsq) < max_r]
    rsq = rsq[np.sqrt(rsq) < max_r]
    if len(rsq)<1:
        continue
    elif np.nanmin(rsq)>1.5**2:
        continue

    # if they pass the check, append them to the lists

    star_rsqs.append(rsq)
    star_fluxes.append(star[fluxkey])
    star_errs.append(star[errkey])

# we need this function for the multiprocessing
# fits the PSF function to the star radial profiles
def get_normed_fluxes(i):
    try:
        p0 = [np.nanmax(star_fluxes[i]), 1.7]
        popt, pcov = curve_fit(fit_psf, np.sqrt(star_rsqs[i]), star_fluxes[i], sigma = star_errs[i], p0=p0)
        
        amp, fwhm = popt[0], popt[1]
        amp_err, fwhm_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
        print(f"finished {i}")
        return amp, fwhm, amp_err, fwhm_err
    except Exception as e:
        print(e)
        return 0,0,0,0


n = len(star_fluxes)

cpu_count = multiprocessing.cpu_count()
print("Number of available CPUs: ", cpu_count)
pool = multiprocessing.Pool(12)

print("Starting to run.")

start = time.time()
endlist = pool.map(get_normed_fluxes, np.arange(0, n))
end = time.time()
print("Time needed: "+str(end-start))

endlist = np.array(endlist)
amps, fwhms, amp_errs, fwhm_errs = endlist[:,0], endlist[:,1], endlist[:,2], endlist[:,3]

# make a cut to keep only reasonable fit results. 1.7 is the FWHM guess.
keepthese = (amps>1)&(amps<300)&(fwhms>1.2)&(fwhms<4.0)&(fwhms!=1.7)

# normalize: r -> r/FWHM and flux -> flux/amp
roverfwhm, stars_normed = [], []
for i in np.where(keepthese)[0]:
    roverfwhm.append(np.sqrt(star_rsqs[i])/fwhms[i])
    stars_normed.append(star_fluxes[i]/amps[i])

# combine all
all_roverfwhm = np.concatenate(roverfwhm)
all_stars_normed = np.concatenate(stars_normed)
order = np.argsort(all_roverfwhm)
all_roverfwhm = all_roverfwhm[order]
all_stars_normed = all_stars_normed[order]

# get the running biweight and its error in small bins
rbins = np.arange(0,5,0.02)
runbiw = []
runbiw_err = []
for rmin, rmax in zip(rbins[:-1], rbins[1:]):
    here = (all_roverfwhm>=rmin)&(all_roverfwhm<rmax)
    runbiw.append(biweight_location(all_stars_normed[here]))
    runbiw_err.append(np.sqrt(biweight_scale(all_stars_normed[here])/(len(here[here]-1))))
runbiw, runbiw_err = np.array(runbiw), np.array(runbiw_err)

if SAVE:
    new_psf = {
        "r/fwhm": rbins[:-1],
        "psf_int": psf_gaus_filt,
        "psf_iter": runbiw,
        "psf_iter_err": runbiw_err
    }
    ascii.write(new_psf, "PSF.tab")