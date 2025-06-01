import numpy as np

# Hartree-Fock Solver for a Minimal Basis Set (e.g., H₂ molecule)

class HartreeFock:
    def __init__(self, S, T, V, ERI, num_electrons):
        """
        Initialize Hartree-Fock solver.
        S   : Overlap matrix (NxN)
        T   : Kinetic energy matrix (NxN)
        V   : Nuclear attraction matrix (NxN)
        ERI : Electron repulsion integrals (NxNxNxN)
        num_electrons: Total number of electrons
        """
        self.S = S
        self.T = T
        self.V = V
        self.ERI = ERI
        self.H_core = T + V
        self.num_electrons = num_electrons
        self.nbf = S.shape[0]
        self.D = np.zeros_like(S)  # Initial density matrix

    def orthogonalize(self):
        """
        Compute orthogonalizer matrix (S^(-1/2)) using symmetric orthogonalization.
        """
        eigvals, eigvecs = np.linalg.eigh(self.S)
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return S_inv_sqrt

    def build_fock(self):
        """
        Build the Fock matrix.
        """
        F = self.H_core.copy()
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                for lam in range(self.nbf):
                    for sig in range(self.nbf):
                        F[mu, nu] += self.D[lam, sig] * (
                            self.ERI[mu, nu, lam, sig] -
                            0.5 * self.ERI[mu, sig, lam, nu]
                        )
        return F

    def scf(self, max_iter=50, convergence=1e-6):
        """
        Self-Consistent Field loop.
        """
        S_inv_sqrt = self.orthogonalize()
        E_total = 0.0

        for iteration in range(1, max_iter + 1):
            F = self.build_fock()

            # Transform Fock matrix to orthonormal basis
            F_prime = S_inv_sqrt.T @ F @ S_inv_sqrt

            # Solve Roothaan equations
            eps, C_prime = np.linalg.eigh(F_prime)

            # Back-transform coefficients
            C = S_inv_sqrt @ C_prime

            # Build new density matrix
            D_new = np.zeros_like(self.D)
            num_occ = self.num_electrons // 2
            for mu in range(self.nbf):
                for nu in range(self.nbf):
                    D_new[mu, nu] = 2 * np.sum(C[mu, :num_occ] * C[nu, :num_occ])

            # Compute electronic energy
            E_elec = np.sum((self.D + D_new) * self.H_core) * 0.5
            E_elec += np.sum((self.D + D_new) * F) * 0.5

            # Check convergence
            delta_D = np.linalg.norm(D_new - self.D)
            print(f"Iteration {iteration}: Energy = {E_elec:.8f}, ΔD = {delta_D:.8e}")

            if delta_D < convergence:
                print("SCF Converged!")
                break

            self.D = D_new
            E_total = E_elec

        return E_total, eps, C

# Example usage (placeholders for S, T, V, ERI)
nbf = 2  # e.g., H₂ with 2 basis functions
num_electrons = 2

# These would be computed with real quantum chemistry integrals
S = np.eye(nbf)
T = np.random.rand(nbf, nbf)
V = np.random.rand(nbf, nbf)
ERI = np.random.rand(nbf, nbf, nbf, nbf)

# Make sure matrices are symmetric
T = 0.5 * (T + T.T)
V = 0.5 * (V + V.T)
ERI = 0.5 * (ERI + ERI.transpose(1,0,2,3))

hf = HartreeFock(S, T, V, ERI, num_electrons)
E_total, eps, C = hf.scf()
print("Final SCF Energy:", E_total)
print("Orbital Energies:", eps)
print("MO Coefficients:\n", C)
