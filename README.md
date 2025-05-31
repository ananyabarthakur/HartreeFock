# Hartree-Fock Solver 

This repository contains a minimal implementation of a **Hartree-Fock (HF) solver** for small molecules in Python using NumPy. It demonstrates the **self-consistent field (SCF) procedure** for computing molecular orbitals and electronic energies.

---

## Overview

- Implements the **restricted Hartree-Fock (RHF)** method.
- Accepts overlap (`S`), kinetic energy (`T`), nuclear attraction (`V`), and electron repulsion (`ERI`) integrals as inputs.
- Supports symmetric orthogonalization and diagonalization of the Fock matrix.
- Builds and updates the density matrix until convergence.
- Prints SCF energies and final molecular orbital coefficients.

---

##  How it Works

1. **Initialization**
   - Provide input matrices (`S`, `T`, `V`, `ERI`) and number of electrons.
2. **SCF Loop**
   - Orthogonalize using symmetric orthogonalization.
   - Build the Fock matrix with Coulomb and exchange terms.
   - Diagonalize the Fock matrix in the orthogonalized basis.
   - Rebuild the density matrix.
   - Iterate until convergence.
3. **Output**
   - Final SCF energy.
   - Orbital energies.
   - Molecular orbital coefficients.

---

##  Example Usage

```python
import numpy as np
from your_hartree_fock_file import HartreeFock

# Example matrices (replace with real integrals for real molecules)
nbf = 2
num_electrons = 2

S = np.eye(nbf)
T = np.random.rand(nbf, nbf)
V = np.random.rand(nbf, nbf)
ERI = np.random.rand(nbf, nbf, nbf, nbf)

# Symmetrize T and V
T = 0.5 * (T + T.T)
V = 0.5 * (V + V.T)
ERI = 0.5 * (ERI + ERI.transpose(1,0,2,3))

hf = HartreeFock(S, T, V, ERI, num_electrons)
E_total, eps, C = hf.scf()

print("Final SCF Energy:", E_total)
print("Orbital Energies:", eps)
print("MO Coefficients:\n", C)

