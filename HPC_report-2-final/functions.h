#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <omp.h>

#include "constants.h"
#include "sparse.h"
#include "read_mat.h"

#include <fstream>
#include <string>

#include <cmath> 


#include <random>


using namespace std;


double scalar_prod(double *v1, double *v2, int nr){
	double sum = 0.0;

	if(nr>par_threshhold){
		#pragma omp parallel for reduction(+:sum)
		for(int i = 0; i < nr; i++){
			sum += v1[i]*v2[i];
		}
		return sum;
	}else{
		for(int i = 0; i < nr; i++){
			sum += v1[i]*v2[i];
		}
		return sum;
	}
	
}

void elementwise_prod(double *v1, double *v2, int nr){
	if(nr>par_threshhold){
		#pragma omp parallel for 
			for(int i = 0; i < nr; i++){
				v1[i] *=v2[i];
			}
	}else{
		for(int i = 0; i < nr; i++){
				v1[i] *=v2[i];
			}
	}
}

double norm(double *v1, int nr){
	// Euclidean norm
	return sqrt(scalar_prod(v1,v1,nr));
}

void normalize(double* v1, int nr){
	double v_norm = norm(v1,nr);
	if(v_norm<eps){
		cout<<"Warning: Norm very small:  "<<endl;
	}else{
		if(nr>par_threshhold){
			#pragma omp parallel for schedule(dynamic, nr/1000+1)
			for(int i=0; i<nr; i++){
				v1[i] /= v_norm;
			}
		}else{
			for(int i=0; i<nr; i++){
				v1[i] /= v_norm;
			}
		}
	}	
}



void vector_update(double *v1, double *p, double alpha, int nr){
	if(nr>par_threshhold){
		#pragma omp parallel for schedule(dynamic, nr/1000+1)
			for(int i = 0; i < nr; ++i){
				v1[i] += alpha * p[i];
			}
		}else{
			for(int i = 0; i < nr; ++i){
				v1[i] += alpha * p[i];
			}
		}
}



void vector_scalarMult(double *v1, double *p, double alpha, int nr){
	if(nr>par_threshhold){
		#pragma omp parallel for schedule(dynamic, nr/1000+1)
		for(int i = 0; i < nr; ++i){
			v1[i] = alpha * p[i];
		}
	}else{
		for(int i = 0; i < nr; ++i){
			v1[i] = alpha * p[i];
		}
	}
}

void vector_add(double alpha,double* v1, double beta, double* v2, double* result, int nr){
	if(nr>par_threshhold){
		#pragma omp parallel for schedule(dynamic, nr/1000+1)
		for(int i = 0; i < nr; ++i){
			result[i] = alpha * v1[i] + beta * v2[i];
		}
	}else{
		for(int i = 0; i < nr; ++i){
			result[i] = alpha * v1[i] + beta * v2[i];
		}
	}
}

void vector_zero_out(double* v, int n){
	if(n>par_threshhold){
		#pragma omp parallel for schedule(dynamic, n/1000+1)
		for(int i = 0; i <n; ++i){
			v[i] = 0;
		}
	}else{
		for(int i = 0; i <n; ++i){
			v[i] = 0;
		}
	}	
}

void vector_zero_out(double* v, int from, int to, int n){
	
if(n>par_threshhold){
	#pragma omp parallel for schedule(dynamic, n/1000+1)
	for(int i = from; i <= to; ++i){
		v[i] = 0;
	}
	}else{
		for(int i = from; i <= to; ++i){
			v[i] = 0;
		}
	}	
}


void mat_zero_out(double** A, int nr, int nc){
	if(nc>par_threshhold){
		#pragma omp parallel for schedule(dynamic, nr/1000+1) 
		for(int i = 0; i <nc; ++i){
			for(int j = 0; j<nr; j++){
				A[i][j] = 0;	
			}	
		}
	}else{
		for(int i = 0; i <nc; ++i){
			for(int j = 0; j<nr; j++){
				A[i][j] = 0;	
			}	
		}
	}
}


void normalize_copy(double *normalized, double *original, int nr){
	double v_norm = norm(original,nr);
	if(v_norm<eps){
		cout<<"Warning: Norm to small "<<v_norm<<endl;
	}else{
		if(nr>par_threshhold){
			#pragma omp parallel for schedule(dynamic, nr/1000+1)
			for(int i=0; i<nr; i++){
				normalized[i] = original[i]/v_norm;
			}
		}else{
			for(int i=0; i<nr; i++){
				normalized[i] = original[i]/v_norm;
			}	
		}
	}
}	


void copyArray(double* dest, const double* src, int size) {
	if(size>par_threshhold){
		    #pragma omp parallel for schedule(dynamic, size/1000+1)
			for(int i=0; i<size; i++){
				dest[i] = src[i];
			}
		}else{
			for(int i=0; i<size; i++){
				dest[i] = src[i];
			}
		}
}

void copyArr2Mat_col(double** matrix, const double* arr, int col, int size){
	// #pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=0; i<size; i++){
		matrix[col][i] = arr[i];
	}
}

void copyArr2Mat_col(double** matrix, const double* arr, int col, int size, int from, int to){
	// #pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=from; i<=to; i++){
		matrix[col][i] = arr[i];
	}
}

void copyArr2Mat_row(double** matrix, const double* arr, int row, int size){
	// #pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=0; i<size; i++){
		matrix[i][row] = arr[i];
	}
}

void copyArr2Mat_row(double** matrix, const double* arr, int row, int size, int from, int to){
	// #pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=from; i<=to; i++){
		matrix[i][row] = arr[i];
	}
}

void copyMat_col2Arr(double** matrix, double* arr, int col, int size){
	// #pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=0; i<size; i++){
		arr[i] = matrix[col][i];
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






void printMat(double** matrix, int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}


void triangularSolver(double** A, double* b, double* x, int n){
	for(int row = n-1; row>=0; row--){
		double sum = 0.0;
		//could be parralelized here or reduction operator
		for(int col = row+1; col<n; col++){
			sum+=A[row][col]*x[col];
		}
		x[row] = (b[row]-sum)/A[row][row];
	}
}




void writeCSV(char* matname, int n, int inner, int iterations,double* b, double* xstar, double* w, double* resvec, double** U, double** R) {
    
	char filename[100];
    strcpy(filename, "data_output/");
    strcat(filename, matname);
    strcat(filename, "_data.csv");
	ofstream file(filename, ofstream::trunc); // Open the file in write mode with truncation


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
    for (int i = 0; i < inner; i++) {
        file << "U_" << i;
        for (int j = 0; j < n; j++) {
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

		// wo muss ich hier freen? oder nicht weil in dekonstruktor gemacht wird?
    } else {
        cout << "FILE NOT OPENED " <<fname<< endl;
    }
}

bool getRHS(char* fname, double* b){
	std::ifstream file(fname);

    if (file.is_open()) {
		double temp_nr;
        int nr;
        file >> temp_nr;
		nr = static_cast<int>(temp_nr);
		for(int i=0;i<nr;i++){
			file >> b[i];
		}
		cout<<"Vector file "<<fname<<endl;
		cout<<"Vector length " << nr<<endl;
		return true;
	}else{
		cout << "FILE NOT OPENED " << fname<< endl;
		return false;
	}
}

void make_b(char* fname,int nr){
	std::ofstream file(fname,std::ofstream::trunc);

	std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
	file<<nr<<endl;
	#pragma omp parallel for schedule(dynamic, nr/1000+1)
	for(int i=0;i<nr;i++){
		 file<<distribution(generator)<<endl;
	}

	cout<<"random Vector of length created " << nr<<endl;
	file.close();
}


void save_solution(char* fname, double* xstar, int n){
	std::ofstream file(fname,std::ofstream::trunc);
	file<<n<<endl;
	for(int i=0;i<n;i++){
		 file<<xstar[i]<<endl;
	}
	cout<<"Vector saved in: " << fname << " length: "<< n<<endl;
	file.close();
}

double* getFEM_sparse(sparse* mat){
	std::ifstream f_coef("FEM_output/coef.txt");
	std::ifstream f_iat("FEM_output/iat.txt");
	std::ifstream f_ja("FEM_output/Ja.txt");
	std::ifstream f_b("FEM_output/b.txt");

	int n_term, n_row_plus1, n_row;
	f_coef >> n_term;
	f_ja >> n_term;
	f_iat >> n_row_plus1;
	f_b >> n_row;

	double* elem = (double*) malloc(n_term*sizeof(double));
	int* ja = (int*) malloc(n_term*sizeof(int));
	int* iat = (int*) malloc(n_row_plus1*sizeof(int));


	double* b = (double*) malloc(n_row*sizeof(double));

	for(int i = 0; i < n_term; i++){
		f_coef >> elem[i];
		f_ja >> ja[i];
	}

	for (int i = 0; i < n_row_plus1; i++){
		f_iat >> iat[i];
	}
	for (int i = 0; i < n_row; i++){
		f_b >> b[i];
	}

	mat->set_nrow(n_row);
	mat->set_ncol(n_row);
	mat->set_n_term(n_term);
	mat->set_iat(iat);
	mat->set_ja(ja);
	mat->set_coef(elem);

	printf("Matrix rows %d columns %d nterm %d\n",n_row,n_row,n_term);
	f_coef.close();
	f_iat.close();
	f_ja.close();
	f_b.close();
	return b;
}

double** mat_allocation(int n_row , int n_col){
	double **A = (double**) malloc(n_row*sizeof(double*));
    for (int i = 0; i < n_row; ++i) {
        A[i] = (double*)malloc(n_col * sizeof(double));
    }
	return A;
}

void mat_free(double** A, int n_col, int n_row){
    for (int i = 0; i < n_row; ++i) {
        free(A[i]);
    }
	free(A);
}


int getNZ(double** A, int n){
	// number of nonzeros in mtx
	// A is square
	int nnz = 0;
	#pragma omp parallel for schedule(dynamic,n/1000+1)
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(abs(A[i][j])>eps){
				#pragma omp atomic
				nnz++;
			}
		}
	}
	return nnz;
}

int getNZ_idx(double** A, double** idx, int n){
	// number of nonzeros in mtx
	// A is square
	int nnz = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(abs(A[i][j])>eps){
				nnz++;
				idx[nnz][0] = i;
				idx[nnz][1] = j;
			}
		}
	}
	return nnz;
}

double max(double* v, int n){
	double maxVal = v[0];
	#pragma omp parallel for schedule(dynamic, n/1000+1)
    for (int i = 1; i < n; i++) {
        if (v[i] > maxVal) {
            #pragma omp critical
            {
                if (v[i] > maxVal) {
                    maxVal = v[i];
                }
            }
        }
    }
    return maxVal;
}

#endif