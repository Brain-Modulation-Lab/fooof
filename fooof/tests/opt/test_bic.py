import fooof
from fooof.objs import FOOOF
from fooof.sim import gen_freqs, gen_power_spectrum
from importlib import reload

from fooof.plts.opt import plot_bic_values
from fooof.plts.error import plot_spectral_error
reload(fooof)
import fooof
print(fooof.__file__)
ap_params = [50, 30,1.5]
gauss_params = [10, 0.3, 2, 15,0.5,4, 20, 0.1, 4, 60, 0.3, 1]
nlv = 0.0025

print(f"real aper params: {ap_params}")
print(f"real gauss params: {gauss_params}")


xs, ys = gen_power_spectrum([1, 150], ap_params, gauss_params, nlv )

tfm = FOOOF(aperiodic_mode='lorentzian-noise-floor',regularization_weight=10**4,bic_opt = True, verbose=False)
tfm.fit(xs, ys)

tfm.report()

plot_bic_values(tfm.models)
