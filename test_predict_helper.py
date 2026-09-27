import sys

sys.path.insert(0, r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\pyTEM')

import numpy as np

from pytem import gerda_io
from pytem.inversion import invert

FDB_PATH = r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Skrivebord\rune.fdb'

sounding = gerda_io.read_sounding(FDB_PATH, dataset_id=228353, position=1)
m = sounding.moments['M1']

n_layers = 10
thicknesses = np.full(n_layers - 1, 5.0)
log_resistivities = np.full(n_layers, np.log(50.0))

result = invert(thicknesses=thicknesses, log_resistivities=log_resistivities,
                 transform='dlf', **gerda_io.invert_kwargs(m))

pred = gerda_io.predict_response(result['thicknesses'], result['resistivities'], m)
print('obs :', m['obs_data'])
print('pred:', pred)
print('rms_history:', result['rms_history'])
