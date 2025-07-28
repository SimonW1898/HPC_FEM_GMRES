# Parallel Finite Element Solver for Convection-Diffusion Equations

## Project Overview

This repository showcases my work on high-performance numerical linear algebra and scientific computing, focused on solving 1D and 2D transient convection-diffusion equations using parallel programming techniques. The centerpiece is an efficient finite element method (FEM) solver for large sparse linear systems, leveraging OpenMP for shared-memory parallelization.

**Main highlights:**
- Implementation of custom sparse matrix data structures (CSR format) for memory-efficient storage and fast computation.
- Assembly of FEM system matrices from mesh and boundary input files.
- Solution of large-scale linear systems via the Generalized Minimal Residual (GMRES) iterative algorithm, including advanced features like Householder projections, Givens rotations, Jacobi preconditioning, and restart strategies.
- Rigorous benchmarking and performance analysis of parallelization efficiency and scalability.
- Comprehensive technical report (LaTeX) documenting theory, design, algorithms, results, and code.

---

## Scientific Problem Tackled

The core mathematical problem is the transient convection-diffusion equation:

**1D:**  
$$-D \frac{\partial^{2} c}{\partial x^{2}} - v_x \frac{\partial c}{\partial x} = \frac{\partial c}{\partial t}$$

**2D:**  
$$\frac{\partial}{\partial x}\left(D_x \frac{\partial c}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial c}{\partial y}\right)- v_x \frac{\partial c}{\partial x} - v_y \frac{\partial c}{\partial y} = \frac{\partial c}{\partial t} + f$$

Discretization via FEM leads to large sparse linear systems, which are solved efficiently and accurately using advanced iterative solvers and parallelization. Boundary conditions and mesh topology are flexibly specified via input files.

---

## File & Folder Structure

- `hpc_report2-2-final/`
    - **main.tex**: Complete technical report (LaTeX) explaining theory, algorithms, implementation, and results.
    - **Figures/**: Plots for convergence, solution fields, and performance metrics.
    - **mesh**, **coord**, **bound**: Example input files for mesh topology, node coordinates, and boundary conditions.

- `src/` or root directory:
    - **main.cpp**: Driver program assembling the FEM problem and invoking the solver.
    - **GMRES.h**: Implementation of the GMRES iterative algorithm, with parallel Householder and Givens routines.
    - **sparse.h**: Custom sparse matrix class in CSR format, including parallel matrix-vector multiplication and preconditioning.
    - **functions.h**, **fem_functions.h**: Core FEM and linear algebra routines.
    - **Other C++ source/header files** supporting the solver and utilities.

---

## Key Contributions & Achievements

- **Advanced Parallelization:** Extensive use of OpenMP for parallel loops, reductions, atomics, and scheduling strategies to maximize shared-memory performance.
- **Custom Sparse Matrix Operations:** Designed and implemented efficient storage and operations for large sparse matrices, crucial for scientific computations.
- **Robust Iterative Solver:** Developed a flexible GMRES solver with state-of-the-art numerical techniques (Householder projections, Givens rotations, Jacobi preconditioning, and restart).
- **Performance Analysis:** Conducted thorough benchmarking, measuring speedup, efficiency, and scalability on large real-world matrices from the SuiteSparse Matrix Collection.
- **Technical Communication:** Authored a comprehensive report detailing the mathematical background, algorithmic strategies, and experimental results.

---

## Technologies Used

- **Languages:** C++ (core implementation), LaTeX (report)
- **Libraries:** OpenMP (parallelization), STL (data structures)
- **Platforms:** Linux/macOS (tested on Intel i5), any system with a C++ compiler and OpenMP support

---

## Report

The detailed project report contains:
- Mathematical derivation of the FEM formulation for convection-diffusion problems.
- Description of the assembly process from input files.
- Algorithmic explanation of the GMRES solver and parallelization strategies.
- Performance results on large-scale benchmarks.
- Figures illustrating convergence, speedup, efficiency, and solution fields.
- Annotated code listings in the appendix.

---

## Author

High Performance Computing & Scientific Programming  
Contact: 

---

## Note

This repository is presented for professional review and demonstration of technical skills in scientific computing, parallel programming, and numerical methods.  
For any questions or further details, feel free to reach out!
