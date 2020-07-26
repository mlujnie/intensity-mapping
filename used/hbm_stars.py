import scipy.stats as stats
import numpy as np
import astropy.units as u
from astropy.io import ascii
from scipy.integrate import quad
import time
import pymc3 as pm
import theano.tensor as tt
import glob

def efunc(z, om, ol):
	return (om*(1+z)**3+ol)**(-1./2)

def integral(z, omega_m, omega_lambda):
	return quad(efunc, 0, z, args=(omega_m, omega_lambda))

def DA(z, h=0.674, omega_m=0.315, omega_lambda=0.685):
	Dh = 3000 * 1/h * u.Mpc 
	return Dh/(1+z)*integral(z, omega_m, omega_lambda)

def kpc_proper_per_deg(z, h=0.674, omega_m=0.315, omega_lambda=0.685):
	return np.pi/(648000)*3600*DA(z, h=0.674, omega_m=0.315, omega_lambda=0.685).to(u.kpc)

def arcsec_to_pkpc(n):
    return kpc_proper_per_deg(2.5)*n/3600. # n arcsec in proper kpc 

def moffat3(xygrid, x0, y0, f0, fwhm):
    x, y = xygrid
    rsq = (x-x0)**2 + (y-y0)**2
    gamma = fwhm/(2 * np.sqrt(2**(1/3.)-1))
    return f0 * (1 + rsq / gamma**2)**(-3)

def moffat_radec(ras, decs, midra, middec, f0, fwhm):
    deccos = np.cos(middec*np.pi/180)
    rsq = ((ras-midra)*deccos)**2 + (decs-middec)**2
    gamma = fwhm/(2 * np.sqrt(2**(1/3.)-1))
    return f0 * (1 + rsq / gamma**2)**(-3)

def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)


####################################################
###__________Simulate the data___________________###
####################################################

np.random.seed(142)
N_stars = 100
N_steps = 6000 # I will need more to get a good convergence. This is just for testing.


# logmean, logstd from stars: 1.2749314016567557 0.7400881322687025
MID_AMP = 1.275
SCALE_AMP = 0.74
STD = 0.05
MID_FWHM = 0.553705981321899    
SCALE_FWHM = 0.21472371245423674 

sigma_x0_real = 0.5/3600  # 0.5 arcseconds


stars = glob.glob("startabs/star_*.tab")
tabs = []
for star in stars[:-1]:
    tab = ascii.read(star)
    tabs.append(tab)

real_ras = []
real_decs = []
guess_ras = []
guess_decs = []
all_ras, all_decs = [], []
for i in range(N_stars):
    _idx = i%len(tabs)
    ras, decs = tabs[_idx]["ra"], tabs[_idx]["dec"]
    midra, middec = np.unique(tabs[_idx]["star_ra"]), np.unique(tabs[_idx]["star_dec"])
    dist = ((midra-ras)*np.cos(middec*np.pi/180.))**2 + (middec-decs)**2
    order = np.argsort(dist)[:40]
    ras, decs = ras[order], decs[order]
    ras, decs = list(ras), list(decs)
    
    star_dec_real = np.unique(tabs[_idx]["star_dec"]) + np.random.normal(loc=0, scale=sigma_x0_real, size=1)
    star_ra_real  = np.unique(tabs[_idx]["star_ra"]) + np.random.normal(loc=0, scale=sigma_x0_real, size=1)/np.cos(star_dec_real*np.pi/180)
    real_ras.append(star_ra_real)
    real_decs.append(star_dec_real)
    
    guess_ras.append(np.unique(tabs[_idx]["star_ra"])[0])
    guess_decs.append(np.unique(tabs[_idx]["star_dec"])[0])
        
    all_ras.append(ras)
    all_decs.append(decs)
        

real_flux = np.exp(np.random.normal(loc=MID_AMP, scale=SCALE_AMP, size=N_stars))
real_fwhm = np.exp(np.random.normal(loc=MID_FWHM, scale=SCALE_FWHM, size=N_stars))/3600.

f_data = []
for i in range(N_stars):
    f_noise = np.random.normal(loc=0, scale=STD, size=real_ras[i].size)
    f_data.append(moffat_radec(all_ras[i], all_decs[i], real_ras[i], real_decs[i], real_flux[i], real_fwhm[i] ) + f_noise)

guess_ras = np.array(guess_ras)
guess_decs = np.array(guess_decs)
all_ras, all_decs = np.array(all_ras), np.array(all_decs)
print("Initiated data.")

PI = np.pi
divnum = (2 * np.sqrt(2**(1/3.)-1))
    
###################################################
###_________Define and run the model____________###
###################################################

with pm.Model() as model:
    
    ##### Priors ###################################
    # priors for fwhm distribution
    mu_fwhm = pm.Uniform("mu_fwhm", 0.1, 1.0)          # mean(log(fwhm/arcsec))
    sigma_fwhm = pm.Uniform("sigma_fwhm", 0.05, 0.5)    # std(log(fwhm/arcsec))
    
    mu_f0 = pm.Uniform("mu_f0", 0.5, 5.0)
    sigma_f0 = pm.Uniform("sigma_f0", 0.001, 1.0)
    
    # priors and values for x0, y0 distributions (make them equal)
    mu_x0 = 0.0 
    sigma_x0 = 0.5 #pm.Uniform("sigma_x0", 0.1, 5)      # in arcsec
   
    log_fwhm_offset = pm.Normal("log_fwhm_offset", mu=0.0, sd=1.0, shape=N_stars) 
    fwhm = pm.Deterministic("fwhm", tt.exp(mu_fwhm + log_fwhm_offset * sigma_fwhm))  
    #fwhm = pm.Deterministic("fwhm", tt.exp(log_fwhm))
    
    f_0_offset = pm.Normal("f_0_offset", mu=0.0, sd=1.0, shape=N_stars)
    f_0 = pm.Deterministic("f_0", tt.exp(mu_f0 + f_0_offset * sigma_f0))   # in 1e-17 erg/s/cm^2
    
    del_ra_offset = pm.Normal("del_ra_offset", mu=0.0, sd=1.0, shape=N_stars)
    del_dec_offset = pm.Normal("del_dec_offset", mu=0.0, sd=1.0, shape=N_stars)

    #del_ra_0 = pm.Normal("del_ra_0", mu=mu_x0, sd=sigma_x0, shape=N_stars)   # I need to change this to /cos(dec)
    #del_dec_0 = pm.Normal("del_dec_0", mu=mu_x0, sd=sigma_x0, shape=N_stars) 

    deccos = tt.cos(guess_decs * PI/180.)
    ra_0 = guess_ras + del_ra_offset * sigma_x0 / 3600. / deccos
    dec_0 = guess_decs + del_dec_offset * sigma_x0 / 3600. 
   
    rsq = tt.stack([((all_ras[i]-ra_0[i])*deccos[i])**2 + (all_decs[i]-dec_0[i])**2 for i in range(N_stars) ])
    gamma = fwhm/divnum
    
    # define model for flux
    flux = tt.stack([f_0[i] * (1 + rsq[i] / gamma[i]**2)**(-3) for i in range(N_stars)])
    # and error
    flux_sd = 0.05
    
    # include observation (likelihood)
    observation = pm.Normal("obs", mu = flux, sd = flux_sd, observed=f_data)
    
    ######### Run, run run!!! #####################
    
    # csv files as backend
    # we have to do this after defining the model
    db = pm.backends.Text('mcmctest')
    trace = pm.sample(N_steps, trace=db) # step=step, 
    burned_trace=trace[1000:]

###################################################
###________Print the results____________________###
###################################################    
end_time = time.time()
print(pm.summary(burned_trace))
print("\n")

print("""## Run 
* **Idea**:  
* **N_stars** = {}
* **grid size** = {} 
* **samples** = {}
* **n_files x file size** = 4 x 
* **warning**:
* **Run time**: {}
* **Notes**: 
    - FWHM: mu:  (real: {:.2f}), sigma:  (real: {:.2f})
    - f_0: mu:  (real: {:.2f}), sigma:  (real {:.2f})
    - sigma_x0:  (real: {:.2f})
* **Interpretation**: """.format(N_stars, all_ras.shape[1], N_steps, pretty_time_delta(end_time-begin_time), MID_FWHM, SCALE_FWHM, MID_AMP, SCALE_AMP, sigma_x0_real))

