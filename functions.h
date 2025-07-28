#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include "constants.h"
#include "sparse.h"
#include "read_mat.h"

#include <fstream>
#include <string>

#include <cmath> 

double scalar_prod(double *v1, double *v2, int nr){
	double sum = 0.0;

	#pragma omp parallel for reduction(+:sum)
	for(int i = 0; i < nr; i++){
		sum += v1[i]*v2[i];
	}
	return sum;
}


double norm(double *v1, int nr){
	// Euclidean norm
	return sqrt(scalar_prod(v1,v1,nr));
}

void normalize(double* v1, int nr){
	double v_norm = norm(v1,nr);
	if(v_norm<eps){
		cout<<"Warning: Norm to small"<<v_norm<<endl;
	}else{
		#pragma omp parallel for
		for(int i=0; i<nr; i++){
			v1[i] /= v_norm;
		}
	}	
}




void vector_update(double *v1, double *p, double alpha, int nr){
	#pragma omp parallel for 
	for(int i = 0; i < nr; ++i){
		v1[i] += alpha * p[i];
	}
}

void vector_scalarMult(double *v1, double *p, double alpha, int nr){
	#pragma omp parallel for 
	for(int i = 0; i < nr; ++i){
		v1[i] = alpha * p[i];
	}
}

void vector_add(double alpha,double* v1, double beta, double* v2, double* result, int nr){
	#pragma omp parallel for 
	for(int i = 0; i < nr; ++i){
		result[i] = alpha * v1[i] + beta * v2[i];
	}
}

void vector_zero_out(double* v, int from, int to, int n){
	#pragma omp parallel for 
	for(int i = from; i <= to; ++i){
		v[i] = 0;
	}
}


void normalize_copy(double *normalized, double *original, int nr){
	double v_norm = norm(original,nr);
	if(v_norm<eps){
		cout<<"Warning: Norm to small "<<v_norm<<endl;
	}else{
		#pragma omp parallel for
		for(int i=0; i<nr; i++){
			normalized[i] = original[i]/v_norm;
		}
	}	

}

void copyArray(double* dest, const double* src, int size) {
    #pragma omp parallel for
    for(int i=0; i<size; i++){
    	dest[i] = src[i];
    }
}

void copyArr2Mat_col(double** matrix, const double* arr, int col, int size){
	#pragma omp parallel for
	for(int i=0; i<size; i++){
		matrix[col][i] = arr[i];
	}
}

void copyArr2Mat_col(double** matrix, const double* arr, int col, int size, int from, int to){
	#pragma omp parallel for
	for(int i=from; i<=to; i++){
		matrix[col][i] = arr[i];
	}
}

void copyArr2Mat_row(double** matrix, const double* arr, int row, int size){
	#pragma omp parallel for
	for(int i=0; i<size; i++){
		matrix[i][row] = arr[i];
	}
}

void copyArr2Mat_row(double** matrix, const double* arr, int row, int size, int from, int to){
	#pragma omp parallel for
	for(int i=from; i<=to; i++){
		matrix[i][row] = arr[i];
	}
}

void copyMat_row2Arr(double** matrix, double* arr, int row, int size){
	#pragma omp parallel for
	for(int i=0; i<size; i++){
		arr[i] = matrix[i][row];
	}
}


double scalarsgn(double x){
	if (x >= 0){
		return 1.0;
	}
	else{
		return -1.0;
	}
}




void printVec(int* v, int nr){
	for(int i = 0; i < nr; i++){
		cout<<v[i]<<"  ";
	}
	cout<<endl;
}


void printVec(double* v, int nr){
	for(int i = 0; i < nr; i++){
		cout<<v[i]<<"  ";
	}
	cout<<endl;
}

void printMat(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}


void triangularSolver(double** A, double* b, double* x, int n){
	for(int row = n-1; row>=0; row--){
		double sum = 0.0;
		for(int col = row+1; col<n; col++){
			sum+=A[row][col]*x[col];
		}
		x[row] = (b[row]-sum)/A[row][row];
	}
}




void writeCSV(int n, int inner, int iterations,double* b, double* xstar, double* w, double* resvec, double** U, double** R) {
    std::ofstream file("data_output/data.csv", std::ofstream::trunc); // Open the file in write mode with truncation
    


	// Write b (1D array)
    file << "b";
    for (int i = 0; i < n; i++) {
        file << "," << b[i];
    }
    file << std::endl;

    // Write xstar (1D array)
    file << "xstar";
    for (int i = 0; i < n; i++) {
        file << "," << xstar[i];
    }
    file << std::endl;
    
    // Write w (1D array)
    file << "w";
    for (int i = 0; i < n + 1; i++) {
        file << "," << w[i];
    }
    file << std::endl;
    
    // Write resvec (1D array)
    file << "resvec";
    for (int i = 0; i < iterations; i++) {
        file << "," << resvec[i];
    }
    file << std::endl;
    
    // Write U (2D array)
    for (int i = 0; i < n; i++) {
        file << "U_" << i;
        for (int j = 0; j < inner; j++) {
            file << "," << U[i][j];
        }
        file << std::endl;
    }
    
    // Write R (2D array)
    for (int i = 0; i < inner; i++) {
        file << "R_" << i;
        for (int j = 0; j < inner; j++) {
            file << "," << R[i][j];
        }
        file << std::endl;
    }
    
    file.close(); // Close the file
}



void createMat(sparse* mat, char* fname) {
    std::ifstream file(fname);

    if (file.is_open()) {
        int nr, nc, nt;
        file >> nr >> nc >> nt;

        int* iat = (int*)malloc((nr + 1) * sizeof(int));
        int* ja = (int*)malloc(nt * sizeof(int));
        double* elem = (double*)malloc(nt * sizeof(double));

        readCSRmat(fname, &nr, &nc, &nt, &iat, &ja, &elem, false);

        mat->set_nrow(nr);
        mat->set_ncol(nc);
        mat->set_n_term(nt);
        mat->set_iat(iat);
        mat->set_ja(ja);
        mat->set_coef(elem);

        file.close();
    } else {
        cout << "FILE NOT OPENED" << endl;
    }
}



#endif