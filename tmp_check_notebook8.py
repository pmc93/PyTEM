import sys
import numpy as np

ROOT = r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\pyTEM"
sys.path.insert(0, ROOT)
from pytem import fwd_circle_central
from pytem.inversion import invert

schematic_thicknesses = np.array([10.0, 12.0])
schematic_resistivities = np.array([300.0, 15.0, 1000.0])
schematic_depths = np.concatenate([[0.0], np.cumsum(schematic_thicknesses)])
schematic_max_depth = schematic_depths[-1] + 20.0

schematic_tx_radius = 30.0
schematic_times = np.logspace(-5, -1, 21)
schematic_dbdt = fwd_circle_central(
    schematic_thicknesses, schematic_resistivities, schematic_tx_radius, schematic_times)

noise_frac = 0.02
rng = np.random.default_rng(1)
d_clean = -schematic_dbdt
d_obs = d_clean * (1.0 + noise_frac * rng.standard_normal(d_clean.size))

n_inv_layers = 25
thick_inv = np.geomspace(1.0, 6.0, n_inv_layers - 1)
thick_inv *= schematic_max_depth / thick_inv.sum()
depths_inv = np.concatenate([[0.0], np.cumsum(thick_inv)])

m0 = np.log(np.full(n_inv_layers, 100.0))
res = invert(
    obs_data=d_obs, thicknesses=thick_inv, log_resistivities=m0,
    tx_size=schematic_tx_radius, times=schematic_times,
    noise_std=noise_frac, alpha_steps=8, maxit=20,
    max_noise_frac=0.0, transform='dlf', geometry='circle_central',
)
smooth_rho = res['resistivities']
print("final RMS:", res['rms_history'][-1], "n_iter:", res['n_iter'])
print("depths_inv:", np.round(depths_inv, 1))
print("smooth_rho:", np.round(smooth_rho, 1))
