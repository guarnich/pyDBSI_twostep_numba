import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, cache=True)
def _build_design_matrix_numba(bvals, bvecs, iso_diffusivities, ax_basis, rad_basis):
    """Costruisce la matrice di design A (Misure x Basi) in modo ottimizzato."""
    n_meas = len(bvals)
    n_aniso = len(bvecs)
    n_iso = len(iso_diffusivities)
    n_total = n_aniso + n_iso
    
    A = np.zeros((n_meas, n_total), dtype=np.float64)
    
    # 1. Basi Anisotropiche (Fibre)
    for j in range(n_aniso):
        # bvecs shape attesa: (N, 3)
        fiber_dir = bvecs[j]
        for i in range(n_meas):
            cos_angle = 0.0
            for k in range(3):
                cos_angle += bvecs[i, k] * fiber_dir[k]
            
            D_app = rad_basis + (ax_basis - rad_basis) * (cos_angle**2)
            A[i, j] = np.exp(-bvals[i] * D_app)
            
    # 2. Basi Isotropiche (Spettro)
    for k in range(n_iso):
        D_iso = iso_diffusivities[k]
        col_idx = n_aniso + k
        for i in range(n_meas):
            A[i, col_idx] = np.exp(-bvals[i] * D_iso)
            
    return A

@jit(nopython=True, fastmath=True, cache=True)
def _solve_nnls_tikhonov_cd(A, y, reg_lambda, max_iter=500, tol=1e-5):
    """Coordinate Descent solver per NNLS con regolarizzazione Tikhonov."""
    n_features = A.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    
    # Pre-calcolo matrici per Tikhonov: (A^T A + lambda*I) w = A^T y
    AtA = A.T @ A
    Aty = A.T @ y
    
    # Aggiungi lambda alla diagonale
    for i in range(n_features):
        AtA[i, i] += reg_lambda

    # Coordinate Descent Loop
    for _ in range(max_iter):
        max_shift = 0.0
        for j in range(n_features):
            old_wj = w[j]
            
            # Calcolo gradiente parziale ottimizzato
            numerator = Aty[j]
            # Sottrai contributo degli altri pesi (dot product sparso)
            for k in range(n_features):
                if k != j:
                    numerator -= AtA[j, k] * w[k]
            
            w_val = numerator / AtA[j, j]
            if w_val < 0: w_val = 0.0 # Proiezione non-negativa
            
            w[j] = w_val
            shift = abs(w[j] - old_wj)
            if shift > max_shift: max_shift = shift
        
        if max_shift < tol: break
            
    return w

@jit(nopython=True, parallel=True, fastmath=True)
def fit_volume_numba(data, bvals, bvecs, mask, iso_grid, ax_basis, rad_basis, reg_lambda, threshold):
    """Loop parallelo sui voxel per lo Step 1."""
    X, Y, Z, N = data.shape
    n_voxels = X * Y * Z
    
    data_flat = data.reshape(n_voxels, N)
    mask_flat = mask.reshape(n_voxels)
    
    # Matrice di design unica
    A = _build_design_matrix_numba(bvals, bvecs, iso_grid, ax_basis, rad_basis)
    
    n_aniso = len(bvecs)
    out_fiber = np.zeros(n_voxels, dtype=np.float64)
    out_restricted = np.zeros(n_voxels, dtype=np.float64)
    out_hindered = np.zeros(n_voxels, dtype=np.float64)
    out_water = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        if not mask_flat[i]: continue
            
        signal = data_flat[i]
        
        # Semplice normalizzazione S0 (media b < 50)
        s0 = 0.0; count = 0
        for k in range(N):
            if bvals[k] < 50:
                s0 += signal[k]; count += 1
        s0 = s0 / count if count > 0 else signal[0]
        
        if s0 <= 1e-6: continue
        y_norm = signal / s0
        
        # Fit
        w = _solve_nnls_tikhonov_cd(A, y_norm, reg_lambda)
        
        # Thresholding (come nel metodo "New")
        w_sum = 0.0
        for k in range(len(w)):
            if w[k] < threshold: w[k] = 0.0
            w_sum += w[k]
            
        if w_sum > 0:
            for k in range(len(w)): w[k] /= w_sum
        else: continue
            
        # Aggregazione metriche
        f_fib = 0.0
        for k in range(n_aniso): f_fib += w[k]
            
        f_res = 0.0; f_hin = 0.0; f_wat = 0.0
        for k in range(len(iso_grid)):
            idx = n_aniso + k
            val = w[idx]
            d = iso_grid[k]
            if d <= 0.3e-3: f_res += val
            elif d <= 2.0e-3: f_hin += val
            else: f_wat += val
                
        out_fiber[i] = f_fib
        out_restricted[i] = f_res
        out_hindered[i] = f_hin
        out_water[i] = f_wat
        
    return out_fiber, out_restricted, out_hindered, out_water