import numpy as np


import pyLIMA.event
import pyLIMA.models.PSPL_model
import pyLIMA.models.USBL_model
import pyLIMA.simulations.simulator
from pyLIMA import telescopes


import matplotlib.pyplot as plt



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
    
    
def astrometric_noise_from_SNR(ra, dec, thetaE, target_SNR=10):
    
    # SNR = 3 marginal, 5 good, 10 strong
    delta_c_max = thetaE / (np.sqrt(2)*2) # u0=sqrt(2)
    rms_astrometry = delta_c_max / target_SNR  # in mas
    
    obs_ra = np.random.normal(ra, rms_astrometry/1000/3600)
    obs_dec = np.random.normal(dec, rms_astrometry/1000/3600)
    
    return obs_ra, obs_dec, rms_astrometry

output_dir = './output/'
roman_positions = np.load("roman_ephemerides.npy")
time = roman_positions[:,0]

t0 = 2458750                     # Time of maximum (HJD)
Ds = 8.                         # Source distance (kpc)
Dl = 2.                          # Lens distance (kpc)
mass = 10.0                       # Lens mass (Solar masses)
pirel = 1/Dl-1/Ds                
thetaE = (8.144*mass*pirel)**0.5 

mu = 8.0                           # Relative proper motion (mas/yr)

# SNR
astrometric_SNR = 10                # Astrometric error parameter


roman_telescope = pyLIMA_telescope_simulation(time)
roman_event =  pyLIMA_event_simulation(roman_telescope, ra=270, dec=-30)


pspl = pyLIMA.models.PSPL_model.PSPLmodel(roman_event,parallax=['Full',t0])

params = [t0, 0.08, thetaE/mu*365.25, thetaE, 1/Ds, -14.901418237131544, -6.013223990947537, -30.0, 270.0, 0.15843220222104726, -0.20748697386621207, 307.3813961361, 1086.2263480900185]
        # [0] Time of maximum, [1] u0 (impact parameter in θ_E)

pyLIMA.simulations.simulator. simulate_lightcurve(pspl,pspl.compute_pyLIMA_parameters(params),add_noise=True)
pyLIMA.simulations.simulator. simulate_astrometry(pspl,pspl.compute_pyLIMA_parameters(params),add_noise=False)


obs_ra,obs_dec = astrometric_noise_from_SNR(roman_telescope.astrometry['ra'].value,roman_telescope.astrometry['dec'].value, thetaE, target_SNR = astrometric_SNR)




lightcurve = np.c_[roman_telescope.lightcurve['time'].value,roman_telescope.lightcurve['flux'].value,roman_telescope.lightcurve['err_flux'].value]

astro = np.c_[roman_telescope.astrometry['time'].value,
              obs_ra,
              [rms_astrometry/1000/3600]*len(obs_ra),
              obs_dec,
              [rms_astrometry/1000/3600]*len(obs_ra),
              ]



roman_telescope2 = telescopes.Telescope('Roman',camera_filter='F146',pixel_scale=0.11,
                                       lightcurve = lightcurve,
                                       lightcurve_names = ['time','flux','err_flux'],
                                       lightcurve_units = ['JD','w/m^2', 'w/m^2'],
                                       astrometry=astro,
                                       astrometry_names = ['time','ra','err_ra','dec','err_dec'],
                                       astrometry_units = ['JD','deg', 'deg','deg','deg'],
                                       location='Space',            spacecraft_name="L2",            spacecraft_positions={
                "astrometry": roman_positions,
                "photometry": roman_positions,
            },
                                       
                                       )

roman_event2 = pyLIMA.event.Event(ra=270, dec=-30)
roman_event2.name = "Roman_" + str(270) + "_" + str(-30)

roman_event2.telescopes.append(roman_telescope2)
pspl2 = pyLIMA.models.PSPL_model.PSPLmodel(roman_event2,parallax=['Full',t0])


from pyLIMA.outputs import pyLIMA_plots


from pyLIMA.fits import TRF_fit,LM_fit,MCMC_fit,DE_fit

trf = TRF_fit.TRFfit(pspl2)
trf.model_parameters_guess = params[:-2]
trf.fit()#computational_pool=pool)


trf.fit_outputs()

plt.show()

cov = trf.fit_results['covariance_matrix']
best = trf.fit_results['best_model']
sample = np.random.multivariate_normal(best,cov,10000)
thetaE_fit = sample[:,3]
piE_fit = np.sqrt(sample[:,-3]**2+sample[:,-4]**2)
mass_fit = thetaE_fit/8.144/piE_fit



breakpoint()
