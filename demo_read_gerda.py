"""
Demo: read one GERDA TEM sounding from rune.fdb and invert it with pyTEM.

Run with the interpreter that has `fdb` installed, e.g.:
    C:\\WPy64-31180\\python-3.11.8.amd64\\python.exe demo_read_gerda.py
"""

import sys

sys.path.insert(0, r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\pyTEM')

import numpy as np

from pytem import gerda_io
from pytem.inversion import invert

FDB_PATH = r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Skrivebord\rune.fdb'

# List a few soundings to pick from.
datasets = gerda_io.list_datasets(FDB_PATH, datatype='tem', limit=10)
print(datasets)

dataset_id = int(datasets['DATASET'].iloc[0])
sounding = gerda_io.read_sounding(FDB_PATH, dataset_id=dataset_id, position=1)

print(f'\nSounding {sounding.dataset_id} ({sounding.ident})')
print(f'  x={sounding.x}, y={sounding.y}, elevation={sounding.elevation}')
print(f'  moments: {list(sounding.moments)}')

for label, m in sounding.moments.items():
    print(f'\n--- moment {label} ---')
    print(f'  n_gates={len(m["times"])}, n_segments_stacked={m["n_segments"]}')
    print(f'  geometry={m["geometry"]}, tx_size={m["tx_size"]}, '
          f'rx_x={m["rx_x"]}, rx_y={m["rx_y"]}')
    print(f'  tx_height={m["tx_height"]}, rx_height={m["rx_height"]}')
    print(f'  peak_current={m["peak_current"]}, tx_turns={m["tx_turns"]}')
    print(f'  times[:5]={m["times"][:5]}')
    print(f'  obs_data[:5]={m["obs_data"][:5]}')
    print(f'  noise_std[:5]={m["noise_std"][:5]}')

# --- Invert the first moment with a simple 4-layer starting model ---
label = next(iter(sounding.moments))
m = sounding.moments[label]

invert_kwargs = {k: v for k, v in m.items() if k in (
    'times', 'obs_data', 'noise_std', 'waveform_times', 'waveform_currents',
    'geometry', 'tx_size', 'rx_x', 'rx_y', 'tx_height', 'rx_height',
)}

n_layers = 10
thicknesses = np.full(n_layers - 1, 5.0)
log_resistivities = np.full(n_layers, np.log(50.0))

result = invert(
    thicknesses=thicknesses,
    log_resistivities=log_resistivities,
    transform='dlf',
    **invert_kwargs,
)

print('\n--- Inversion result ---')
print('Final resistivities:', np.exp(result['log_resistivities']))
print('RMS:', result.get('rms'))
