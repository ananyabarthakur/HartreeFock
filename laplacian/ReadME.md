# 3D Finite Difference Laplacian and Kinetic Energy Operator

This repository contains a simple Python script that demonstrates **how to approximate the Laplacian operator using finite differences** on a **3D grid**. The Laplacian is essential in quantum mechanics, especially for computing the **kinetic energy** of a particle.

---

##  What Does This Code Do?

- Creates a **3D grid** representing a small region of space.  
- Builds a **sample wavefunction** on that grid—a 3D Gaussian (a smooth, bell-shaped function).  
-Approximates the **Laplacian operator** using finite differences.  
-Applies the **kinetic energy operator**:
\[
\hat{T}\psi = -\frac{1}{2} \nabla^2 \psi
\]
- Visualizes:
- The wavefunction itself (a slice in the z=0 plane).
- The kinetic energy operator applied to that wavefunction.

---

## Why Use This?

This example is a **mini quantum simulation** that shows:
- How to discretize the Laplacian on a grid.
- How to apply the kinetic energy operator to a wavefunction.
- A foundation for building more complex quantum mechanics simulations like Schrödinger or Hartree-Fock solvers.

---

##  How to Run

1. Install dependencies:
```bash
pip install numpy matplotlib
