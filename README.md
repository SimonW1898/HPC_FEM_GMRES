# HPC_FEM_GMRES: High-Performance FEM and GMRES Solver for Convection-Diffusion Problems

## Project Overview

This repository demonstrates efficient numerical solution techniques for 1D and 2D transient convection-diffusion equations using the Finite Element Method (FEM) and a custom GMRES iterative solver. The project focuses on sparse matrix storage, parallelization with OpenMP, and robust linear algebra routines, with all core logic implemented in C++ from scratch.

**All numerical algorithms, matrix structures (CSR), and iterative solvers are written without external scientific libraries. Parallelization is achieved using OpenMP, and the implementation is designed for extensibility and scientific benchmarking.**

**Highlights:**
- Custom sparse matrix (CSR) class and efficient matrix-vector multiplication.
- FEM system assembly from mesh and boundary input files.
- GMRES solver with Householder projections, Givens rotations, and Jacobi preconditioning.
- OpenMP parallelization across matrix and solver routines.
- Comprehensive technical report included (theory, design, experiments, annotated code).

---

## Scientific Problem Tackled

This project addresses the numerical solution of the transient convection-diffusion equation:

### 1D Form
$$
-D \frac{\partial^2 c}{\partial x^2} - v_x \frac{\partial c}{\partial x} = \frac{\partial c}{\partial t}
$$

### 2D Form
$$
\frac{\partial}{\partial x}\left(D_x \frac{\partial c}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial c}{\partial y}\right) - v_x \frac{\partial c}{\partial x} - v_y \frac{\partial c}{\partial y} = \frac{\partial c}{\partial t}
$$

where $D$, $v_x$, and $v_y$ are diffusion and convection coefficients. FEM discretization leads to large sparse linear systems, solved efficiently with GMRES and parallelization.

---

## Main Components

### 1. FEM and Sparse Matrix Modules

- **CSR Sparse Matrix Structure:**  
  Custom implementation for memory-efficient storage and fast arithmetic.
- **Matrix Assembly:**  
  FEM system matrix built from mesh and boundary files (text format).
- **Core Linear Algebra:**  
  All matrix, vector, and FEM routines written in C++ for transparency and performance.

### 2. GMRES Iterative Solver

- **Robust GMRES Algorithm:**  
  Householder projections, Givens rotations, Jacobi preconditioning, and restart mechanics.
- **Parallelization:**  
  Heavy use of OpenMP for parallel loops, reductions, and atomic operations across matrix-vector products and solver steps.

### 3. Analysis and Benchmarking

- **Performance Studies:**  
  Benchmark results for speedup and scalability on large matrices (SuiteSparse Collection).
- **Documentation:**  
  Full technical report (LaTeX/PDF) covering mathematical background, algorithmic details, results, and code walkthrough.

---

## File Structure

```
HPC_FEM_GMRES/
├── Report.pdf                # Complete technical report (LaTeX)
├── main.cpp                  # Main driver: problem setup, matrix assembly, solver invocation
├── GMRES.h                   # GMRES algorithm, Householder/Givens/Jacobi routines
├── sparse.h                  # CSR sparse matrix implementation
├── functions.h               # Core FEM and linear algebra utilities
├── fem_functions.h           # FEM assembly and boundary routines
├── [Other .cpp/.h files]     # Supporting code and utilities
└── README.md                 # Project documentation
```

---

## Dependencies

- **C++ (core implementation)**
- **OpenMP** (parallelization)
- **Standard Template Library (STL)**
- **LaTeX** (for report compilation)

Tested on Linux/macOS (Intel i5/i7). Requires C++ compiler with OpenMP support.

---

## Usage

To compile and run:
```bash
g++ -O3 -fopenmp main.cpp -o fem_solver
./fem_solver
```
Configure mesh and boundary files as described in the report.

---

## Known Issues & Limitations

- Only shared-memory parallelization (OpenMP) is implemented; distributed-memory (MPI) is not supported.
- Mesh and boundary input formats are simple text files (see report for specifications).
- Extension to higher dimensions or more element types requires further development.

---

## Contributing

Contributions welcome, especially for:
- Alternative preconditioners or solver optimizations
- MPI or domain decomposition support
- Mesh generation and visualization tools
- Performance tuning and code refactoring

---

## License

This project is for academic research. Please cite appropriately if used in publications.

---

## Author

**[Simon W](https://www.linkedin.com/in/simon-w-32183a292)**  

---

## Note

This repository is presented for professional review and demonstration of technical skills in scientific computing, parallel programming, and numerical PDE solvers.  
For questions, suggestions, or collaboration inquiries, contact me via LinkedIn or GitHub!