from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .twostep import DBSI_TwoStep
from .twostep_numba import DBSI_TwoStep_Numba
from .common import DBSIParams
from .utils import load_dwi_data_dipy, save_parameter_maps, estimate_snr

# Imposta il modello Numba come default per performance
DBSIModel = DBSI_TwoStep_Numba

__version__ = "0.3.0_numba"
print("DBSI Toolbox v0.3.0 (Numba Accelerated) loaded.")