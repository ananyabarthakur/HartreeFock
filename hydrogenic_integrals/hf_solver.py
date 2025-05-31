import numpy as np

# Integrals for H2 (R=0.74 Å, STO-3G) — approximate values

S = np.array([[1.0, 0.6593],
              [0.6593, 1.0]])

T = np.array([[0.7600, 0.2365],
              [0.2365, 0.7600]])

V = np.array([[-1.2266, -0.5974],
              [-0.5974, -1.2266]])

# Core Hamiltonian
H_core = T + V

# Two-electron integrals in physicist's notation (μν|λσ)
ERI = np.zeros((2,2,2,2))

ERI[0,0,0,0] = 0.7746
ERI[0,0,1,1] = 0.5697
ERI[1,1,0,0] = 0.5697
ERI[1,1,1,1] = 0.7746
ERI[0,1,0,1] = 0.4441
ERI[0,1,1,0] = 0.4441

# Symmetrize ERI
ERI[0,0,1,1] = ERI[0,0,1,1]
ERI[1,1,0,0] = ERI[1,1,0,0]
ERI[0,1,1,0] = ERI[0,1,0,1]
ERI[1,0,0,1] = ERI[0,1,0,1]
ERI[1,0,1,0] = ERI[0,1,0,1]
ERI[0,1,0,1] = ERI[0,1,0,1]

num_electrons = 2
nbf = 2

# Orthogonalizer (S^(-1/2))
eigvals, eigvecs = np.linalg.eigh(S)
S_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T

# Initialize density matrix
D = np.zeros((nbf, nbf))

max_iter = 50
convergence = 1e-6

print("Starting Hartree-Fock SCF calculation...")

for iteration in range(1, max_iter+1):
    # Build Fock matrix
    F = H_core.copy()
    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    F[mu, nu] += D[lam, sig] * (
                        ERI[mu, nu, lam, sig] -
                        0.5 * ERI[mu, sig, lam, nu]
                    )

    # Transform Fock to orthonormal basis
    F_prime = S_inv_sqrt.T @ F @ S_inv_sqrt

    # Solve Roothaan equations
    eps, C_prime = np.linalg.eigh(F_prime)

    # Back-transform to original basis
    C = S_inv_sqrt @ C_prime

    # Build new density matrix
    D_new = np.zeros((nbf, nbf))
    num_occ = num_electrons // 2
    for mu in range(nbf):
        for nu in range(nbf):
            D_new[mu, nu] = 2 * np.sum(C[mu, :num_occ] * C[nu, :num_occ])

    # Electronic energy
    E_elec = np.sum((D_new * (H_core + F)))
    delta_D = np.linalg.norm(D_new - D)

    print(f"Iteration {iteration}: E_elec = {E_elec:.8f}  ΔD = {delta_D:.8e}")

    if delta_D < convergence:
        print("SCF Converged!")
        break

    D = D_new

print(f"\nFinal SCF Energy: {E_elec:.8f} Hartree")
print(f"Orbital Energies: {eps}")
print("MO Coefficients:\n", C)
