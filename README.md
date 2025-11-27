
# pyDBSI_twostep_numba: Numba-Accelerated DBSI Toolbox

A high-performance Python toolbox for fitting the **Diffusion Basis Spectrum Imaging (DBSI)** model. This version leverages **Numba JIT compilation** and parallel processing to drastically reduce computation time while maintaining the mathematical robustness of the "Two-Step" approach.

## 🚀 Performance

| Step | Standard (Scipy) | Accelerated (Numba) | Speedup |
| :--- | :---: | :---: | :---: |
| **Step 1 (Spectral Fit)** | ~20 min / volume | **~30 sec / volume** | **~40x** |
| **Step 2 (Tensor Refinement)** | Full Volume | Targeted (Fiber Mask) | Optimized |

## 🧠 What is DBSI?

**Diffusion Basis Spectrum Imaging (DBSI)** resolves complex tissue microstructures by modeling the diffusion signal as a combination of:
1.  **Anisotropic Tensors:** Organized structures like axonal fibers.
2.  **Isotropic Spectrum:** A range of diffusivities representing cells, edema, and CSF [Wang, Y. et al., 2011].

### The "Two-Step" Approach (Accelerated)

1.  **Step 1 (Spectral - Accelerated):** Uses a custom **Numba-based Coordinate Descent solver** to solve the Non-Negative Least Squares (NNLS) problem with Tikhonov regularization in parallel across all voxels.
2.  **Step 2 (Tensor - Hybrid):** Refines axial/radial diffusivities using bounded Non-Linear Least Squares (Scipy), applied selectively only to voxels where fibers were detected in Step 1.

-----

## 📦 Installation

This package requires a Python environment with `numba` installed.

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/guarnich/pyDBSI_twostep_numba.git](https://github.com/guarnich/pyDBSI_twostep_numba.git)
    cd pyDBSI_twostep_numba
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the Toolbox:**

    ```bash
    pip install -e .
    ```

-----

## ⚡ Quickstart

Run the complete pipeline (SNR estimation -> Calibration -> Accelerated Fitting) using the command-line interface:

```bash
python examples/run_dbsi.py \
    --input "subject/dwi/dwi_preproc.nii.gz" \
    --bval  "subject/dwi/dwi.bval" \
    --bvec  "subject/dwi/dwi.bvec" \
    --mask  "subject/dwi/brain_mask.nii.gz" \
    --out   "subject/dbsi_results_numba"
````

### Advanced Usage

You can control the solver parameters directly:

```bash
python examples/run_dbsi.py \
    --input "data.nii.gz" \
    --bval "data.bval" \
    --bvec "data.bvec" \
    --mask "mask.nii.gz" \
    --out  "results/" \
    --snr 30.0 \
    --mc_iter 2000
```

-----

## 🛠️ Key Differences from Standard Implementation

  * **Dependency:** Requires `numba` \>= 0.55.0.
  * **Solver:** Replaces `scipy.optimize.nnls` with a JIT-compiled Coordinate Descent solver.
  * **Precision:** Uses `float64` for internal solver arithmetic to ensure numerical stability equivalent to Scipy.
  * **Thresholding:** Applies consistent sparsity filtering (default 1%) in both linear and non-linear steps to ensure physical plausibility.

## 📚 References

  * **Wang, Y. et al. (2011).** *Quantification of increased cellularity during inflammatory demyelination.* Brain.
  * **Cross, A.H. & Song, S.K. (2017).** *A new imaging modality to non-invasively assess multiple sclerosis pathology.* J Neuroimmunol.

<!-- end list -->

```
```