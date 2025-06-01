import numpy as np
from pyscf import gto, scf, ao2mo

def hartree_fock(molecule_xyz, basis='4-31g', max_iter=50, convergence=1e-6):
    """
    Hartree-Fock SCF solver using PySCF for integrals and 4-31G basis set.
    """
    # 1. Define molecule
    mol = gto.Mole()
    mol.atom = molecule_xyz
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.build()

    # 2. Get integrals from PySCF
    S = mol.intor('int1e_ovlp')      # Overlap matrix
    T = mol.intor('int1e_kin')       # Kinetic energy
    V = mol.intor('int1e_nuc')       # Nuclear attraction
    H_core = T + V

    eri = mol.intor('int2e')         # 2-electron repulsion integrals (chemists' notation)

    nbf = S.shape[0]                 # Number of basis functions
    nelec = mol.nelectron            # Number of electrons

    # 3. Orthogonalizer (S^-1/2)
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # 4. Initialize density matrix
    D = np.zeros((nbf, nbf))

    print(f"Starting RHF with {nelec} electrons and {nbf} basis functions...")

    for iteration in range(1, max_iter+1):
        # 5. Build Fock matrix
        J = np.zeros((nbf, nbf))
        K = np.zeros((nbf, nbf))
        for mu in range(nbf):
            for nu in range(nbf):
                for lam in range(nbf):
                    for sig in range(nbf):
                        J[mu, nu] += D[lam, sig] * eri[mu, nu, lam, sig]
                        K[mu, nu] += D[lam, sig] * eri[mu, sig, lam, nu]
        F = H_core + J - 0.5 * K

        # 6. Transform Fock matrix to orthonormal basis
        F_prime = S_inv_sqrt.T @ F @ S_inv_sqrt

        # 7. Solve Roothaan equations
        eps, C_prime = np.linalg.eigh(F_prime)
        C = S_inv_sqrt @ C_prime

        # 8. Build new density matrix
        D_new = np.zeros((nbf, nbf))
        nocc = nelec // 2
        for mu in range(nbf):
            for nu in range(nbf):
                D_new[mu, nu] = 2 * np.sum(C[mu, :nocc] * C[nu, :nocc])

        # 9. Electronic energy
        E_elec = np.sum((D_new * (H_core + F)) * 0.5)
        delta_D = np.linalg.norm(D_new - D)

        print(f"Iteration {iteration}: E_elec = {E_elec:.8f}  ΔD = {delta_D:.8e}")

        if delta_D < convergence:
            print("SCF Converged!")
            break

        D = D_new

    # 10. Add nuclear repulsion
    E_nuc = mol.energy_nuc()
    E_total = E_elec + E_nuc

    print(f"\nTotal Energy: {E_total:.8f} Hartree")
    print("Orbital Energies (Hartree):", eps)
    print("MO Coefficients:\n", C)

    return E_total, eps, C

if __name__ == "__main__":
    # Example molecule: Water
    molecule_xyz = '''
    O 0.000000 0.000000 0.000000
    H 0.000000 -0.757160 0.586260
    H 0.000000 0.757160 0.586260
    '''
    hartree_fock(molecule_xyz)
