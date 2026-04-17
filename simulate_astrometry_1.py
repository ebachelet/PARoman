import numpy as np
import matplotlib.pyplot as plt


import pyLIMA.event
import pyLIMA.models.PSPL_model
import pyLIMA.models.USBL_model
import pyLIMA.simulations.simulator
from pyLIMA import telescopes
import pandas as pd

from pyLIMA.outputs import pyLIMA_plots

from pyLIMA.fits import TRF_fit,LM_fit,MCMC_fit,DE_fit

import matplotlib.pyplot as plt

def compute_chi2(telescope, pspl_model, best_params):
    pyLIMA_params = pspl_model.compute_pyLIMA_parameters(best_params)
    model_output  = pspl_model.compute_the_microlensing_model(telescope, pyLIMA_params)
    
    # Photometric chi2
    flux_model = model_output['photometry']
    flux_obs   = telescope.lightcurve['flux'].value
    err_flux   = telescope.lightcurve['err_flux'].value
    chi2_phot  = np.sum(((flux_obs - flux_model) / err_flux) ** 2)
    
    # Astrometric chi2 — pyLIMA returns shape (2, N): row 0 = ra, row 1 = dec
    astro_model = model_output['astrometry']
    ra_model  = astro_model[0]
    dec_model = astro_model[1]
    
    ra_obs  = telescope.astrometry['ra'].value
    dec_obs = telescope.astrometry['dec'].value
    err_ra  = telescope.astrometry['err_ra'].value
    err_dec = telescope.astrometry['err_dec'].value
    
    chi2_ra    = np.sum(((ra_obs  - ra_model)  / err_ra)  ** 2)
    chi2_dec   = np.sum(((dec_obs - dec_model) / err_dec) ** 2)
    chi2_astro = chi2_ra + chi2_dec
    
    return chi2_phot, chi2_astro

def refine_microlensing_parameters(microlensing_parameters, time):
    # Refine parameters to distribution towards the bulge, especially tE

    new_ml_parameters = np.copy(microlensing_parameters)

    new_t0 = t0_in_Roman_windows(time, microlensing_parameters[0])

    new_ml_parameters[0] = new_t0

    new_tE = 10 ** (np.random.normal(1.2, 0.3))
    new_ml_parameters[2] = new_tE

    mag_source = np.random.uniform(15, 25)
    f_source = 10 ** ((27.4 - mag_source) / 2.5)
    g_blend = np.random.uniform(0, 2)

    new_ml_parameters[-2] = f_source
    new_ml_parameters[-1] = f_source * g_blend

    return new_ml_parameters


def t0_in_Roman_windows(time, t0):
    windows_length = 80

    tstart = time[0]
    tend = tstart + windows_length

    windows = []

    for i in range(6):
        mask = (time < tend) & (time > tstart)

        windows.append(time[mask])

        try:
            tstart = time[np.where(mask)[0][-1] + 1]
            tend = tstart + windows_length

        except:
            pass

    choosen_one = np.random.choice([0, 1, 2, 3, 4, 5])

    window = windows[choosen_one]

    t0 = np.random.uniform(window[0] + 5, window[-1] - 5)  # +/-5 day safety

    return t0


def pyLIMA_telescope_simulation(time):
    try:
        roman_positions = np.load("roman_ephemerides.npy")
        roman = pyLIMA.simulations.simulator.simulate_a_telescope(
            "Roman",
            timestamps=time,pixel_scale=0.11,
            location="Space",
            spacecraft_name="L2",
            spacecraft_positions={
                "astrometry": roman_positions,
                "photometry": roman_positions,
            },
            astrometry=True,
        )

    except:
        roman = pyLIMA.simulations.simulator.simulate_a_telescope(
            "Roman",
            timestamps=time,pixel_scale=0.11,
            location="Space",
            spacecraft_name="L2",
            astrometry=True,
        )
        roman.initialize_positions()
        np.save(
            "roman_ephemerides.npy", roman.spacecraft_positions["photometry"]
        )

    return roman


def pyLIMA_event_simulation(roman_telescope, ra=270, dec=-30):
    roman_event = pyLIMA.event.Event(ra=ra, dec=dec)
    roman_event.name = "Roman_" + str(ra) + "_" + str(dec)

    roman_event.telescopes.append(roman_telescope)

    return roman_event


def simulate_microlensing_PSPL(roman_telescope, resolution=50, ra=270, dec=-30):

    params_limits = [[roman_telescope.lightcurve_flux['time'].value.min(),
                      roman_telescope.lightcurve_flux['time'].value.max()], 
                      [-1,1],
                      [1,100],
                      [15,25],
                      [15,25]]
    events_grid = construct_the_hyper_grid(params_limits,resolution)

    roman_event = pyLIMA_event_simulation(roman_telescope, ra=ra, dec=dec)
    breakpoint()
    pspl = pyLIMA.models.PSPL_model.PSPLmodel(roman_event)
    
    lcs = []
    
    
    for pspl_parameters in events_grid:

        pyLIMA_parameters = pspl.compute_pyLIMA_parameters(pspl_parameters)

        magnification = pspl.model_magnification(roman_telescope, pyLIMA_parameters)

        flux = pspl_parameters[-2] * magnification + pspl_parameters[-1]
        
        flux_obs,eflux_obs =  noise_model(flux, exptime=50)
        
        lcs.append([flux_obs,eflux_obs])
        
    return lcs,events_grid
    
    
def astrometric_noise(ra,dec, level = 5): #mas


    obs_ra = np.random.normal(ra,level/1000/3600)
    obs_dec = np.random.normal(dec,level/1000/3600)


    return obs_ra,obs_dec

output_dir = './output/'
roman_positions = np.load("roman_ephemerides.npy")
time = roman_positions[:,0]

t0 = 2458750                     # Time of maximum (HJD)
Ds = 8.                         # Source distance (kpc)
Dl = 3.                          # Lens distance (kpc)
#mass = 30                       # Lens mass (Solar masses)
#pirel = 1/Dl-1/Ds                
#thetaE = (8.144*mass*pirel)**0.5 

mu = 6                           # Relative proper motion (mas/yr)

rms_astrometry = 0.15                # Astrometric error (mas)

# Mass grid
mass_grid = np.logspace(np.log10(3), np.log10(1000), 20)

N_trials = 20

results_mass_study = []

# Build telescope
roman_telescope = pyLIMA_telescope_simulation(time)


for mass in mass_grid:
    print(f"\nSimulating M = {mass:.1f} M☉")
    
    pirel = 1/Dl - 1/Ds
    thetaE = (8.144 * mass * pirel)**0.5
    tE = thetaE/mu * 365.25
    print('tE = ', tE)
    print('thetaE = ', thetaE)

    for trial in range(N_trials):
        u0_trial = np.random.uniform(0.1, 0.3)
        rms_trial = np.random.uniform(0.1, 0.2)
        phi = np.random.uniform(0, 2*np.pi)
        piE_N = (pirel / thetaE) * np.cos(phi)
        piE_E = (pirel / thetaE) * np.sin(phi)
        t0_trial = t0_in_Roman_windows(time, t0)
        params = [t0_trial, 
                  u0_trial, 
                  tE, 
                  thetaE, 
                  1/Ds, 
                  piE_N, 
                  piE_E, 
                  -30.0, 
                  270.0, 
                  0.15843220222104726, 
                  -0.20748697386621207, 
                  307.3813961361, 
                  1086.2263480900185]
        

        roman_event = pyLIMA_event_simulation(roman_telescope, ra=270, dec=-30)
        pspl = pyLIMA.models.PSPL_model.PSPLmodel(roman_event, parallax=['Full', t0])

        pyLIMA.simulations.simulator.simulate_lightcurve(pspl, pspl.compute_pyLIMA_parameters(params), add_noise=True)
        pyLIMA.simulations.simulator.simulate_astrometry(pspl, pspl.compute_pyLIMA_parameters(params), add_noise=False)

        obs_ra, obs_dec = astrometric_noise(
            roman_telescope.astrometry['ra'].value,
            roman_telescope.astrometry['dec'].value,
            level=rms_trial
        )

        lightcurve = np.c_[
            roman_telescope.lightcurve['time'].value,
            roman_telescope.lightcurve['flux'].value,
            roman_telescope.lightcurve['err_flux'].value
        ]
        astro = np.c_[
            roman_telescope.astrometry['time'].value,
            obs_ra,  [rms_trial/1000/3600] * len(obs_ra),
            obs_dec, [rms_trial/1000/3600] * len(obs_dec)
        ]

        roman_telescope2 = telescopes.Telescope(
            'Roman', camera_filter='F146', pixel_scale=0.11,
            lightcurve=lightcurve,
            lightcurve_names=['time','flux','err_flux'],
            lightcurve_units=['JD','w/m^2','w/m^2'],
            astrometry=astro,
            astrometry_names=['time','ra','err_ra','dec','err_dec'],
            astrometry_units=['JD','deg','deg','deg','deg'],
            location='Space', spacecraft_name="L2",
            spacecraft_positions={"astrometry": roman_positions, "photometry": roman_positions}
        )

        roman_event2 = pyLIMA.event.Event(ra=270, dec=-30)
        roman_event2.name = "Roman_270_-30"
        roman_event2.telescopes.append(roman_telescope2)
        pspl2 = pyLIMA.models.PSPL_model.PSPLmodel(roman_event2, parallax=['Full', t0])

        try:
            trf = TRF_fit.TRFfit(pspl2)
            trf.model_parameters_guess = params[:-2]
            trf.fit_parameters['tE'][1]  = [1, 5000]
            trf.fit_parameters['theta_E'][1] = [0.1, 200]
            trf.fit_parameters['piEN'][1] = [-2, 2]
            trf.fit_parameters['piEE'][1] = [-2, 2]
            trf.fit()

            best = trf.fit_results['best_model']
            cov  = trf.fit_results['covariance_matrix']

            chi2_phot, chi2_astro = compute_chi2(roman_telescope2, pspl2, best)
            chi2_total = chi2_phot + chi2_astro

            sample  = np.random.multivariate_normal(best, cov, 10000)
            thetaE_fit = np.abs(sample[:, 3])
            piE_fit = np.sqrt(sample[:, 5]**2 + sample[:, 6]**2)
            valid = piE_fit > 0
            mass_samples = np.abs(sample[valid, 3]) / (8.144 * piE_fit[valid])
            mass_samples = mass_samples[(mass_samples > 0) & (mass_samples < 1e5)] # clip

            mass_recovered = np.median(mass_samples)
            mass_error     = np.std(mass_samples)

            # Threshold levels relative to chi2_total
            chi2_10pct  = chi2_total * 0.10   # 10% of measured chi2
            chi2_100pct = 160.0               # fixed detection threshold
            chi2_200pct = 320.0               # twice detection threshold

            event_id = f"M{mass:.1f}_T{trial:03d}"

            results_mass_study.append({
                'event_id':         event_id,
                'mass_input':       mass,
                'mass_recovered':   mass_recovered,
                'sigmaM':           mass_error,
                'u0':               u0_trial,
                'rms_ast':          rms_trial,
                'thetaE':           thetaE,
                'tE':               tE,
                'chi2_total':       chi2_total,
                'chi2_phot':        chi2_phot,
                'chi2_astro':       chi2_astro,
                'chi2_10pct':       chi2_10pct,
                'chi2_100pct':      chi2_100pct,
                'chi2_200pct':      chi2_200pct,
                'detected':         chi2_total > chi2_100pct,
                'strong_detection': chi2_total > chi2_200pct,
                'confirmed_BH':     mass_recovered - 3*mass_error > 3.0
            })

        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            import traceback
            traceback.print_exc()

            results_mass_study.append({
                'event_id':         f"M{mass:.1f}_T{trial:03d}",
                'mass_input':       mass,
                'mass_recovered':   np.nan,
                'sigmaM':           np.nan,
                'u0':               u0_trial,
                'rms_ast':          rms_trial,
                'thetaE':           thetaE,
                'tE':               tE,
                'chi2_total':       np.nan,
                'chi2_phot':        np.nan,
                'chi2_astro':       np.nan,
                'chi2_10pct':       np.nan,
                'chi2_100pct':      160.0,
                'chi2_200pct':      320.0,
                'detected':         False,
                'strong_detection': False,
                'confirmed_BH':     False
            })

            print(list(trf.fit_parameters.keys()))

df_mass = pd.DataFrame(results_mass_study)
df_mass.to_csv('results_mass.csv', index=False)
print(df_mass)



