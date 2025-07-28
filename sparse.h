#ifndef SPARSE_H
#define SPARSE_H

#include <omp.h>
#include <iostream>

class sparse{
private:
	int nrow, ncol, n_term;
	double* coef; // non zero matrixelements
	int *iat, *ja; // first non zero column index iat, what is ja?


public:
	// constructors
	sparse(){};
	sparse(int nrow, int ncol, int n_term);
	~sparse(); // destructor


	// get
	int get_nrow(){return nrow;}
	int get_ncol(){return ncol;}
	int get_n_term(){return n_term;}

	double* get_coef(){return coef;}
	int* get_iat(){return iat;}
	int* get_ja(){return ja;}


	// set
	void set_nrow(int n){nrow = n;}
	void set_ncol(int n){ncol = n;}
	void set_n_term(int n){n_term = n;}

	// void alloc_coef(int n){coef = (double*)malloc((n)*sizeof(double));}
	// void alloc_iat(int n){iat = (int*)malloc((n+1)*sizeof(int));}
	// void alloc_ja(int n){ja = (int*)malloc((n)*sizeof(int));}

	void set_coef(double* v){coef = v;}
	void set_iat(int* v){iat = v;}
	void set_ja(int* v){ja = v;}


	// matrix funcitons
	void post_MV(double* v, double* y); // A*b
	void pre_MV(double* v, double* y); // b'*A
	void matrixProduct(sparse* A_ptr, sparse* result); 
	// display mtx
	void printMat();
};


using namespace std;

sparse :: sparse(int nr, int nc, int nt){
	nrow = nr;
	ncol = nc;
	n_term = nt;

	coef = (double*)malloc(nt*sizeof(double));
	ja = (int*)malloc(nt*sizeof(int));
	iat = (int*)malloc(nr*sizeof(int));
};

sparse :: ~sparse(){
	free(coef);
	free(ja);
	free(iat);
}


void sparse::post_MV(double* v, double* y ){
	// #pragma omp for schedule(dynamic,nrow/1000+1)
	for(int i=0; i<nrow; i++){
		double sum = 0.0;
		#pragma omp simd reduction(+:sum)
		for(int j=iat[i]; j<iat[i+1]; j++){
			sum += coef[j]*v[ja[j]];
		}
		y[i] = sum;
	}
}

void sparse::pre_MV(double* v, double* y ){
	#pragma omp for schedule(dynamic,nrow/1000+1)
	for(int j=0; j<ncol; j++){
		double sum = 0.0;
		// #pragma omp simd reduction(+:sum)
		for(int i=0; i<nrow; i++){
			for(int k=iat[i]; k<iat[i+1]; k++){
				if (ja[k] == j){
					sum += v[i]*coef[k];
				}
			}
		}
		y[j] = sum;
	}
}


void sparse::matrixProduct(sparse* B_ptr, sparse* result){
	// (nxm)*(mxk) = (nxk) but usually square anyways


	sparse& B = *B_ptr;

	if (nrow != B.ncol){
		cout<<"Incompatible Size"<<endl;
	}


	for(int i = 0; i < nrow; i++){
		double sum = 0.0;
		for(int j = 0; j < B.ncol; j++){

			sum += i+j;
		}


	}


}







#include <iomanip>

void sparse::printMat() {
    // Create a 2D matrix to represent the sparse matrix
    double** matrix = new double*[nrow];
    for (int i = 0; i < nrow; i++) {
        matrix[i] = new double[ncol];
        // Initialize all elements to 0
        for (int j = 0; j < ncol; j++) {
            matrix[i][j] = 0.0;
        }
    }

    // Fill the matrix with the non-zero elements from the sparse matrix
    for (int row = 0; row < nrow; row++) {
        int start = iat[row];                  // Starting index for the current row
        int end = iat[row + 1];                // Ending index for the current row

        // Iterate over the non-zero elements in the current row
        for (int index = start; index < end; index++) {
            int column = ja[index];             // Column index of the non-zero element
            double value = coef[index];         // Value of the non-zero element
            matrix[row][column] = value;        // Store the value in the matrix
        }
    }

    // Print the matrix
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            cout << setw(8) << matrix[i][j] << " ";
        }
        cout << endl;
    }

    // Free the memory allocated for the matrix
    for (int i = 0; i < nrow; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}










// void gmres(sparse* A, double* b, double* x){
// 	int nc = A.get_ncol();
// 	int nr = A.get_nrow();
// 	if(nr!=nc){
// 		cout<<"ERROR nr != nc"<<endl;
// 	}
// 	double tol = 1e-6;
// 	int maxit = min(nr,10) // 10 is small
// 	double n2b = norm(b,n);
// 	double* minv_b = (double*)malloc((nc)*sizeof(double));
// 	vector_update(minv_b,b,1,nc);
// 	int inner = maxit;
// 	int outer = maxit;

// 	int flag = 1;
// 	double* xmin = (double*)malloc((nr)*sizeof(double));
// 	vector_update(xmin,b,1,nc);
// 	int imin = 0;
// 	int jmin = 0;
// 	double tolb = tol*n2b;
// 	int evalxm = 0;
// 	int stag = 0;
// 	int moresteps = 0;
// 	int maxmsteps = 3;
// 	int minupdated = 0;


// 	double r = (double*)malloc((nr)*sizeof(double));
// 	double Ax = (double*)malloc((nr)*sizeof(double));
// 	A.postMV(x,Ax);
// 	vector_add(1.,b,-1.,Ax,r,nc);



// 	free(minv_b);
// 	free(xmin);
// 	free(r);
// 	free(Ax);
// }





















#endif