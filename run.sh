#export CPATH=/opt/homebrew/include
#export LIBRARY_PATH=/opt/homebrew/lib

#!/bin/bash

# Set the number of threads
export OMP_NUM_THREADS=4

g++-12 main.cpp -o main -fopenmp -std=c++11 -O2 -o main.out
time ./main.out
#-larmadillo



PYTHON_ENV="/Users/simonwenchel/Studium/Semester04/HPC/MatrixOperations/env/bin/python3"

"$PYTHON_ENV" evaluate.py