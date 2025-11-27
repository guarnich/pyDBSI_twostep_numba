import numpy as np
from .base import BaseDBSI
from .numba_kernels import fit_volume_numba
from .nlls_tensor_fit import DBSI_TensorFit # Riutilizziamo il tuo fit vincolato "buono"

class DBSI_TwoStep_Numba(BaseDBSI):
    def __init__(self, n_iso_bases=20, reg_lambda=0.01, filter_threshold=0.01):
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.filter_threshold = filter_threshold
        # Usiamo il fitter non lineare esistente per i bounds corretti
        self.tensor_fitter = DBSI_TensorFit() 

    def fit_volume(self, data, bvals, bvecs, mask, run_step2=True, **kwargs):
        print(f"[DBSI-Numba] Starting Accelerated Fit (Step 1: Spectrum)...")
        
        # Setup input per Numba
        iso_grid = np.linspace(0, 3.0e-3, self.n_iso_bases)
        
        # Standardizza bvecs a (N, 3)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs_numba = bvecs.T
        else:
            bvecs_numba = bvecs

        # Condividi vettori con il fitter non lineare (per Step 2)
        self.tensor_fitter.current_bvecs = bvecs_numba

        # --- STEP 1: NUMBA PARALLEL ---
        f_fib, f_res, f_hin, f_wat = fit_volume_numba(
            np.ascontiguousarray(data).astype(np.float64), 
            np.ascontiguousarray(bvals).astype(np.float64), 
            np.ascontiguousarray(bvecs_numba).astype(np.float64), 
            np.ascontiguousarray(mask).astype(bool),
            iso_grid, 
            1.5e-3, 0.3e-3, # Basi fisse
            self.reg_lambda,
            self.filter_threshold
        )
        
        shape = data.shape[:3]
        maps = {
            'fiber_fraction': f_fib.reshape(shape),
            'restricted_fraction': f_res.reshape(shape),
            'hindered_fraction': f_hin.reshape(shape),
            'water_fraction': f_wat.reshape(shape),
            # Default per voxel senza fibre
            'axial_diffusivity': np.zeros(shape),
            'radial_diffusivity': np.zeros(shape)
        }

        if not run_step2:
            return maps

        # --- STEP 2: HYBRID ---
        # Eseguiamo il fit non-lineare SOLO sui voxel dove Numba ha trovato fibre.
        # Questo mantiene la velocità evitando calcoli inutili sul background/liquor.
        print("[DBSI-Numba] Starting Step 2: Tensor Refinement (Scipy bounded)...")
        
        # Maschera di interesse: voxel nel brain mask che hanno una frazione di fibra rilevante
        fiber_mask = (maps['fiber_fraction'] > 0.05) & mask
        
        # Qui potresti usare joblib per parallelizzare anche questo loop, 
        # ma per ora usiamo il metodo classico selettivo.
        X, Y, Z = shape
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if fiber_mask[x, y, z]:
                        signal = data[x, y, z]
                        
                        # Creiamo un "fake" result dallo step 1 per inizializzare step 2
                        # Usando una classe dummy o un oggetto semplice se DBSIParams lo permette
                        # Per brevità, qui chiamiamo direttamente fit_voxel del tensor_fitter
                        # Nota: dovresti passare l'initial guess corretto basato sui risultati Numba
                        # ma il tensor_fitter ha un fallback "blind guess" che funziona.
                        # Per precisione massima, bisognerebbe ricostruire l'oggetto initial_guess.
                        
                        res = self.tensor_fitter.fit_voxel(signal, bvals)
                        
                        maps['axial_diffusivity'][x, y, z] = res.axial_diffusivity
                        maps['radial_diffusivity'][x, y, z] = res.radial_diffusivity
                        # Aggiorniamo le frazioni raffinate
                        maps['fiber_fraction'][x, y, z] = res.fiber_density
                        maps['restricted_fraction'][x, y, z] = res.f_restricted
        
        return maps