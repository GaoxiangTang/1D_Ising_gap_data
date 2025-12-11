# Reproducing Results for arXiv:2512.03526

This repository contains the source code and data used to reproduce the results and figures in **arXiv:2512.03526**.

The work proves the **stretched exponential scaling of parity-restricted energy gaps** in a **1D random transverse-field Ising model (RTIM)**. The numerical experiments consist of two main parts:

1. **Exact diagonalization of the single-particle Hamiltonian** We use exact diagonalization of the single-particle Hamiltonian to obtain the excitation energies of free fermions, which are then mapped to the energy gaps of the Ising models. See, for example, Eq. (8) of *Physical Review B 106, 064204 (2022)* for details.  
   - Time complexity: $\Theta(L^3)$, where $L$ is the number of sites.  
   - Numerical precision: limited by dense SVD precision, approximately $10^{-13}$.

2. **Analytical upper bound on the parity-restricted energy gap** We compute the upper bound on the parity-restricted energy gaps of 1D RTIMs derived in this work. See Supplemental Material, Part II, for details. The implementation is written in **Julia** for performance and high precision.
   - Time complexity: $\Theta(L^2)$ or $\Theta(L^3)$, depending on the calculation mode.  
   - Numerical precision: can reach approximately $10^{-150}$ (using Julia's arbitrary precision arithmetic).

---

## Repository Structure

### `src/` – Source code for the main routines

- `exact_diagonalization.py`  
  Performs exact diagonalization of the single-particle Hamiltonian to obtain the excitation energies of free fermions and map them to Ising energy gaps.

- `GapBound.jl` & `gap_bound_parallel.jl`  
  - `GapBound.jl` serves as the module interface that encapsulates the core logic.  
  - `gap_bound_parallel.jl` calculates the bound for a **specific system size** $L$. It supports two modes: a complete scan of all possible partitions, or a sampled scan (skipping intervals) for faster approximations.

- `gap_bound_parallel_equal_spacing.jl`  
  Designed to efficiently calculate bounds for a series of **equally spaced system sizes**. It calls `GapBound.jl` but improves performance by reusing partition data from smaller systems (sub-structures) when calculating larger systems.

- `fitting.py`  
  Performs statistical analysis of the stretched exponential scaling behavior.

### `figures/`

Figures included in the article **arXiv:2512.03526**.

---

## Data

The data required to reproduce figures in this work can be found in [https://doi.org/10.5281/zenodo.17890439](https://doi.org/10.5281/zenodo.17890439).

## Notebooks

- `exact_diagonalization.ipynb`  
  Demonstration of how to use `exact_diagonalization.py` to generate data.

- `fig1.ipynb`, `fig2.ipynb`, `fig3.ipynb`  
  Notebooks to reproduce Figures 1–3 in the article using the provided data.
