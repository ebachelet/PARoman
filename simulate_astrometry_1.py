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

from multiprocessing import Pool

import matplotlib.pyplot as plt

def compute_chi2(telescope, pspl_model, best_params):
    pyLIMA_params = pspl_model.compute_pyLIMA_parameters(best_params)
    model_output  = pspl_model.compute_the_microlensing_model(telescope, pyLIMA_params)
    
    # Photometric chi2
    flux_model = model_output['photometry']
    flux_obs = telescope.lightcurve['flux'].value
    err_flux = telescope.lightcurve['err_flux'].value
    chi2_phot = np.sum(((flux_obs - flux_model) / err_flux) ** 2)
    
    # Astrometric chi2 — pyLIMA returns shape (2, N): row 0 = ra, row 1 = dec
    astro_model = model_output['astrometry']
    ra_model  = astro_model[0]
    dec_model = astro_model[1]
    
    ra_obs  = telescope.astrometry['ra'].value
    dec_obs = telescope.astrometry['dec'].value
    err_ra  = telescope.astrometry['err_ra'].value
    err_dec = telescope.astrometry['err_dec'].value

    # Temp
    # print(f"  ra_model range: {ra_model.min():.6f} to {ra_model.max():.6f}")
    # print(f"  ra_obs range: {ra_obs.min():.6f} to {ra_obs.max():.6f}")
    # print(f"  err_ra sample:  {err_ra[0]:.2e}")
    
    chi2_ra = np.sum(((ra_obs  - ra_model)  / err_ra)  ** 2)
    chi2_dec = np.sum(((dec_obs - dec_model) / err_dec) ** 2)
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

    for i in range(10):
        mask = (time < tend) & (time > tstart)

        windows.append(time[mask])

        try:
            tstart = time[np.where(mask)[0][-1] + 1]
            tend = tstart + windows_length

        except:
            pass

    choosen_one = np.random.choice(range(10))

    window = windows[choosen_one]

    t0 = np.random.uniform(window[0] + 5, window[-1] - 5)  # +/-5 day safety

    return t0

def build_roman_time():
    # Time scale matching Roma's realistic cadence
    # 6 high-cadence seasons of 72 days at 12.1-min sampling
    # 4 low-cadence seasons of 72 days at 3-day sampling
    cadence_high = 12.1 / (60.0 * 24.0)
    cadence_low  = 3.0
    year = 365.25

    # years 1 and 2
    s1 = np.arange(0, 72, cadence_high)
    s2 = np.arange(year*0.5, year*0.5 + 72, cadence_high)
    s3 = np.arange(year*1.0, year*1.0 + 72, cadence_high)

    # year 3
    s4 = np.arange(year*2.0, year*2.0 + 72, cadence_low)
    s5 = np.arange(year*2.5, year*2.5 + 72, cadence_low)
    s6 = np.arange(year*3.0, year*3.0 + 72, cadence_low)
    s7 = np.arange(year*3.5, year*3.5 + 72, cadence_low)

    # years 4 and 5
    s8  = np.arange(year*4.0, year*4.0 + 72, cadence_high)
    s9  = np.arange(year*4.5, year*4.5 + 72, cadence_high)
    s10 = np.arange(year*5.0, year*5.0 + 72, cadence_high)

    return np.concatenate([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]) + 2458750.0


def pyLIMA_telescope_simulation(time):
    try:
        roman_positions = np.load("roman_ephemerides.npy")
        roman = pyLIMA.simulations.simulator.simulate_a_telescope(
            "Roman",
            timestamps=time, pixel_scale=0.11,
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
            timestamps=time, pixel_scale=0.11,
            location="Space",
            spacecraft_name="L2",
            astrometry=True,
        )
        roman.initialize_positions()
        roman_positions = roman.spacecraft_positions["photometry"]
        np.save("roman_ephemerides.npy", roman_positions)
 
    return roman, roman_positions


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
#roman_positions = np.load("roman_ephemerides.npy")
#time = roman_positions[:,0]
time = build_roman_time()


t0 = 2458750                     # Time of maximum (HJD)
Ds = 8.                         # Source distance (kpc)
year = 365.25

# Mass grid
mass_grid = np.logspace(np.log10(3), np.log10(100), 100)

N_trials = 4

results_mass_study = []

# Build telescope
roman_telescope, roman_positions = pyLIMA_telescope_simulation(time)


for mass in mass_grid:
    print(f"\nSimulating M = {mass:.1f} M☉")

    for trial in range(N_trials):
        u0_trial = np.random.uniform(-1, 1)

        mu = np.random.normal(6, 1)                 # Relative proper motion (mas/yr)

        # Dl
        Dl = np.random.uniform(1, Ds- 0.1)
        pirel = 1/Dl - 1/Ds
        
        thetaE = (8.144 * mass * pirel)**0.5
        tE = thetaE/mu * 365.25
        print('tE = ', tE)
        print('thetaE = ', thetaE)

        #fb

        mag_source = np.random.triangular(16, 24, 25)
        fsource = 10**((27.4 - mag_source) / 2.5)
        ftotal = fsource + # fb

        
        phi = np.random.uniform(0, 2*np.pi)
        piE_N = (pirel / thetaE) * np.cos(phi)
        piE_E = (pirel / thetaE) * np.sin(phi)
        piE_true = np.sqrt(piE_N**2 + piE_E**2)

        t0_trial = t0_in_Roman_windows(time, t0)

        mu_N = (piE_N / piE_true) * (thetaE / tE) * 365.25
        mu_E = (piE_E / piE_true) * (thetaE / tE) * 365.25

        print(f"piE_true = {piE_true:.4f}")
        #print(f" Expected astro shift: {thetaE:.2f} mas, noise: {rms_trial:.2f} mas")  

        season_starts = 2458750 + np.array([0, year * 0.5, year * 1.0, year * 2.0, year * 2.5, year * 3.0, year * 3.5, year * 4.0, year * 4.5, year * 5.0, ])
        
        params = [t0_trial,
                    u0_trial,
                    tE,
                    thetaE,
                    1/Ds,
                    mu_N,      # piE_N old
                    mu_E,      # piE_E old
                    -30.0,      # position_N
                    270.0,      # position_E
                    piE_N,       # mu_N old
                    piE_E,       # mu_E old
                    fsource,
                    ftotal]
              

        roman_event = pyLIMA_event_simulation(roman_telescope, ra=270, dec=-30)
        pspl = pyLIMA.models.PSPL_model.PSPLmodel(roman_event, parallax=['Full', t0])

        pyLIMA.simulations.simulator.simulate_lightcurve(pspl, pspl.compute_pyLIMA_parameters(params), add_noise=True)
        pyLIMA.simulations.simulator.simulate_astrometry(pspl, pspl.compute_pyLIMA_parameters(params), add_noise=False)

        # SNR and error in magnitude
        #rms_phot  = np.sqrt(fsource + 100)        #  noise = signal + background

        flux_obs = roman_telescope.lightcurve['flux'].value
        err_flux = roman_telescope.lightcurve['err_flux'].value

        em = (2.5 / np.log(10)) * (err_flux / flux_obs)
        max_em = np.nanmax(em)
        snr_phot = 1.0 / max_em

        # RMS
        psf_fwhm_mas = 110.0
        rms_trial = (psf_fwhm_mas * max_em)
        rms_trial = np.clip(rms_trial, 1, 100.0)

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
            te_factor = np.clip(1.0 + max_em * 10, 1.5, 5.0)
            trf.fit_parameters['tE'][1] = [tE / te_factor, tE * te_factor]
            trf.fit_parameters['tE'][1] = [tE * 0.5, tE * 2.0]
            trf.fit_parameters['theta_E'][1] = [0.1, 200]
            trf.fit_parameters['piEN'][1] = [-0.5, 0.5]
            trf.fit_parameters['piEE'][1] = [-0.5, 0.5]
            trf.fit_parameters['u0'][1] = [-1, 1]
            trf.fit_parameters['fsource_Roman'][1] = [1.0, 1e5]
            trf.fit_parameters['ftotal_Roman'][1]  = [0.0, 1e5]
            trf.model_parameters_guess = params[:11]
            trf.fit()

            print(f" Guess: {params[:-2]}")
            print(f" piEN bounds: {trf.fit_parameters['piEN']}")
            # print(f" Guess length: {len(params[:-2])}, n params: {len(trf.fit_parameters)}")

            ################################################

            #from pyLIMA.outputs import pyLIMA_plots

            #from pyLIMA.fits import TRF_fit,LM_fit,MCMC_fit,DE_fit

            #trf = TRF_fit.TRFfit(pspl2)
            #trf.model_parameters_guess = params[:-2]
            #trf.fit()#computational_pool=pool)

            #trf.fit_outputs()

            #plt.show()
            
            #breakpoint()

            ################################################

            best = trf.fit_results['best_model']
            cov = trf.fit_results['covariance_matrix']

            # Diagnostic
            piE_best = np.sqrt(best[9]**2 + best[10]**2)
            mass_naive = best[3] / (8.144 * piE_best)
            print(f"  thetaE_fit={best[3]:.3f}, piE_fit={piE_best:.4f}, naive_mass={mass_naive:.2f}")
            print(f"  thetaE_true={thetaE:.3f}, piE_true={piE_true:.4f}, true_mass={mass:.2f}")

            if np.any(np.diag(cov) < 0):
                print(f"  Warning: bad covariance at M={mass:.1f}, trial={trial}")

            chi2_phot, chi2_astro = compute_chi2(roman_telescope2, pspl2, best)
            chi2_total = chi2_phot + chi2_astro

            print(f"  chi2_phot={chi2_phot:.0f}  chi2_astro={chi2_astro:.0f}  ratio={chi2_astro/chi2_total:.2%}")

            #############################################################################################################
            ## TRF block

            #thetaE_best = best[3]
            #piE_best = np.sqrt(best[9]**2 + best[10]**2)
            #mass_recovered = thetaE_best / (8.144 * piE_best)

            ## Diagonal covariance errors
            #sig_thetaE = np.sqrt(abs(cov[3, 3]))
            #sig_piEN = np.sqrt(abs(cov[9, 9]))
            #sig_piEE = np.sqrt(abs(cov[10, 10]))
            #sig_piE = np.sqrt((best[9]*sig_piEN)**2 + (best[10]*sig_piEE)**2) / piE_best
            #mass_error = mass_recovered * np.sqrt((sig_thetaE/thetaE_best)**2 + (sig_piE/piE_best)**2)


            #print(f"  mass_recovered={mass_recovered:.2f}, mass_error={mass_error:.2f}, true={mass:.2f}")

            #breakpoint()

            ##############################################################################################################
            # MCMC block

            mcmc = MCMC_fit.MCMCfit(pspl2)
            mcmc.model_parameters_guess = best
            mcmc.MCMC_links = 1000      # total steps per walker (min:1000)
            mcmc.MCMC_walkers = 2      # number of walkers (min:2)

            with Pool(processes=8) as pool:
                mcmc.fit(computational_pool=pool)

            chains = mcmc.fit_results['MCMC_chains']  # shape: (n_walkers, n_steps, n_params)
            flat = chains[:, chains.shape[1]//2:, :].reshape(-1, chains.shape[2])

            thetaE_samples = flat[:, 3]
            piE_samples = np.sqrt(flat[:, 9]**2 + flat[:, 10]**2)
            valid = (piE_samples > 0) & (thetaE_samples > 0)
            mass_samples = thetaE_samples[valid] / (8.144 * piE_samples[valid])

            mass_recovered = np.median(mass_samples)
            mass_error = np.std(mass_samples)
            print(f"  MCMC mass_recovered={mass_recovered:.2f} +/- {mass_error:.2f}, true={mass:.2f}")

            breakpoint()

            ################################################################################################################

            # Threshold levels relative to chi2_total
            chi2_10pct = chi2_total * 0.10   # 10% of measured chi2
            chi2_100pct = 160.0               # fixed detection threshold
            chi2_200pct = 320.0               # twice detection threshold

            event_id = f"M{mass:.1f}_T{trial:03d}"

            results_mass_study.append({
                'event_id':         event_id,
                'mass_input':       mass,
                'mass_recovered':   mass_recovered,
                'sigmaM':           mass_error,
                'mu':               mu,
                'mu_N':             mu_N,
                'mu_E':             mu_E,
                'pirel':            pirel,
                'Ds':               Ds,
                'Dl':               Dl,
                'piE_true':         piE_true,
                'max_em':           max_em,
                'snr_phot':         snr_phot, 
                'fsource':          fsource,    
                'mag_source':       mag_source,
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
                'confirmed_BH':     mass_recovered - 3*mass_error > 3.0,
                'tE_in_survey':     tE < 72 * 6            # Avoid events with tE bigger than 72
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
                'mu':               mu,
                'mu_N':             mu_N,
                'mu_E':             mu_E,
                'pirel':            pirel,
                'Ds':               Ds,
                'Dl':               Dl,
                'piE_true':         piE_true,
                'max_em':           np.nan,
                'snr_phot':         np.nan, 
                'fsource':          fsource,   
                'mag_source':       mag_source,
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
                'confirmed_BH':     False,
                'tE_in_survey':     tE < 72 * 6,            # Avoid events with tE bigger than 72
            })

            print(list(trf.fit_parameters.keys()))
            print(f" Guess length: {len(params[:-2])}, params keys: {len(trf.fit_parameters)}")

df_mass = pd.DataFrame(results_mass_study)
df_mass.to_csv('results_mass.csv', index=False)
print(df_mass)





