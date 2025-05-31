What’s Happening
- Hard-coded realistic H₂ integrals from literature.

- Used an orthogonalizer (S⁻¹ᐟ²) to handle overlap.

- Built the Fock matrix with Coulomb and Exchange terms.

- Diagonalized the Fock matrix in the orthogonalized basis.

- Rebuilt the density matrix until convergence.

 Extensions
- Add nuclear repulsion term (for H₂ it’s ~0.7140 Hartree at 0.74 Å).

- Implement analytic gradients (forces) for geometry optimization.

- Expand to larger molecules with more integrals (PySCF or Psi4).

