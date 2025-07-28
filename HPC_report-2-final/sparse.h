#ifndef SPARSE_H
#define SPARSE_H

#include <omp.h>
#include <iostream>
#include <cmath>

#include "constants.h"

class sparse{
private:
	int nrow, ncol, n_term;
	double* coef; // non zero matrixelements
	int *iat, *ja; // first non zero column index iat, what is ja?


public:
	// constructors
	sparse();
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
	void set_nrow(int n){
		nrow = n;
		ncol = n;
		free(iat);
		iat = (int*) malloc(ncol*sizeof(int));
	}
	void set_ncol(int n){
		nrow = n;
		ncol = n;
		free(iat);
		iat = (int*) malloc(ncol*sizeof(int));
	}
	void set_n_term(int n){
		n_term = n;
		free(coef);
		free(ja);
		ja = (int*) malloc(n_term*sizeof(int));
		coef = (double*) malloc(n_term*sizeof(double));
	}

	// void alloc_coef(int n){coef = (double*)malloc((n)*sizeof(double));}
	// void alloc_iat(int n){iat = (int*)malloc((n+1)*sizeof(int));}
	// void alloc_ja(int n){ja = (int*)malloc((n)*sizeof(int));}

	void set_coef(double* v){coef = v;}
	void set_iat(int* v){iat = v;}
	void set_ja(int* v){ja = v;}

	void copy_coef(double* v){
		#pragma omp parallel for
		for(int i = 0; i<n_term;i++){
			coef[i]=v[i];
		}
	}

	void copy_iat(int* v){
		#pragma omp parallel for
		for(int i = 0; i<nrow+1;i++){
			iat[i]=v[i];
		}
	}

	void copy_ja(int* v){
		#pragma omp parallel for
		for(int i = 0; i<n_term;i++){
			ja[i]=v[i];
		}
	}


	//operators
	sparse& operator=(sparse&);
	friend bool operator==(const sparse& lhs, const sparse& rhs);

	//Data Operations
	void full2sparse(double** A, int n_row);


	// matrix funcitons
	void addition_update(sparse* B_ptr);
	void scalarMult(double alpha);
	void post_MV(double* v, double* y); // A*b
	void pre_MV(double* v, double* y); // b'*A
	void matrixProduct(sparse* A_ptr, sparse* result); 
	void diag(double* v);
	void getJacobi(double* v);
	void diag_x_sparse(double* diag);
	double* left_Jacobi();


	// display mtx
	void printMat();
	void printComponents();
};


using namespace std;

sparse::sparse(){
	// cout<<"empty constructor\n";
	nrow = 0;
	ncol = 0;
	n_term = 0;

	coef = nullptr;
	ja = nullptr;
	iat = nullptr;
};

sparse :: sparse(int nr, int nc, int nt){
	nrow = nr;
	ncol = nc;
	n_term = nt;

	coef = (double*)malloc(nt*sizeof(double));
	ja = (int*)malloc(nt*sizeof(int));
	iat = (int*)malloc((nr+1)*sizeof(int));	//+1
};

sparse :: ~sparse(){
	free(coef);
	free(ja);
	free(iat);
}


void sparse::post_MV(double* v, double* y ){
	// #pragma omp parallel for schedule(dynamic,nrow/1000+1)
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
	#pragma omp parallel for schedule(dynamic,nrow/1000+1)
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

sparse& sparse::operator=(sparse& A){
	// std::cout<<"cpy"<<endl;
	int n_row = A.get_nrow();
	int n_term = A.get_n_term();

	this->set_n_term(n_term);
	this->set_ncol(n_row);
    this->copy_coef(A.get_coef());
    this->copy_iat(A.get_iat());
    this->copy_ja(A.get_ja());

    return *this;
}

bool operator==(const sparse& lhs, const sparse& rhs) {
    if (lhs.nrow != rhs.nrow || lhs.ncol != rhs.ncol || lhs.n_term != rhs.n_term) {
		cout<<" Shapes are not equal\n";
		return false; 

    }

    // Compare ja arrays
    for (int i = 0; i < lhs.n_term; i++) {
        if (lhs.ja[i] != rhs.ja[i]) {
			cout<<" ja arrays are not equal\n";
			return false; // 

        }
    }

    // Compare iat arrays
    for (int i = 0; i <= lhs.nrow; i++) {
        if (lhs.iat[i] != rhs.iat[i]) {
			cout<<" iat arrays are not equal\n";
			return false; // 

        }
    }

    return true; // Shapes, ja, and iat arrays are equal
}


void sparse::full2sparse(double** A, int n){ 
	// A is square
	// A is sparse --> suppose only max 10% of elements are nonzero
	double sparse_ratio = 1;
	int n_term_max = static_cast<int>(n * n * sparse_ratio);
	
	double* coef_temp = (double*)malloc(n_term_max*sizeof(double));
	int* Ja_temp = (int*)malloc(n_term_max*sizeof(int));
	int* iat_temp = (int*)malloc((n+1)*sizeof(int));

	this->set_ncol(n);
	this->set_nrow(n);

	int idx_nt = 0;
	int idx_nr = 0;
	bool new_row = true;

	#pragma omp parallel for schedule(dynamic,n/1000+1)
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(abs(A[i][j])>eps){
				coef_temp[idx_nt] = A[i][j];
				Ja_temp[idx_nt] = j;
				if(new_row){
					iat_temp[idx_nr] = idx_nt;
 					idx_nr++;
					new_row = false;
				}
				idx_nt++;
			}
		}
		new_row = true;
	}
	iat_temp[idx_nr] = idx_nt;

	this->set_n_term(idx_nt);
	this->copy_coef(coef_temp);
	this->copy_ja(Ja_temp);
	this->copy_iat(iat_temp);

	free(iat_temp);
	free(coef_temp);
	free(Ja_temp);
}

void sparse::addition_update(sparse* B_ptr){
	if(*this==*B_ptr){
		double* add = B_ptr->get_coef();
		#pragma omp parallel for schedule(dynamic,n_term/1000 +1)
		for(int i = 0; i<n_term; i++){
			coef[i] += add[i];
		}
	}else{
		cout<<"Sparsity structure is not the same \n";
	}

}


void sparse::scalarMult(double alpha){
	#pragma omp parallel for schedule(dynamic,n_term/1000 +1)
	for(int i = 0; i<n_term; i++){
		coef[i] *= alpha;
	}
}
void sparse::matrixProduct(sparse* B_ptr, sparse* result){
	// (nxm)*(mxk) = (nxk) but usually square anyways


	sparse& B = *B_ptr;

	if (nrow != B.ncol){
		std::cout<<"Incompatible Size"<<endl;
	}


	for(int i = 0; i < nrow; i++){
		double sum = 0.0;
		for(int j = 0; j < B.ncol; j++){

			sum += i+j;
		}


	}


}


void sparse::diag(double* v){
	int start, end;
	#pragma omp parallel for schedule(dynamic, ncol/1000+1)
	for(int i = 0; i<ncol; i++){
		start = iat[i];
        end = iat[i+1];
        for(int j = start; j<end; j++){
			if(ja[j] == i){
				v[i] = coef[j];
				break;
			}
		}
	}
}

void sparse::getJacobi(double* v){
	this->diag(v);
	int zeros = 0;
	#pragma omp parallel for schedule(dynamic, ncol/1000 +1)
	for(int i = 0; i<ncol; i++){
		if (v[i] == 0){
			v[i]=1;
			#pragma omp atomic
			zeros++;
			}
		v[i] = 1/v[i];
	}
	if(zeros>0){
		cout<< "WARNING: Number of zeros in diagonal: "<<zeros<<endl;
	}
}

void sparse::diag_x_sparse(double* diag){
	int start, end;
	#pragma omp parallel for schedule(dynamic, ncol/1000+1)
	for(int i = 0; i<ncol; i++){
		start = iat[i];
        end = iat[i+1];
        for(int j = start; j<end; j++){
			coef[j] *= diag[i]; 

		}
	}
}


double* sparse::left_Jacobi(){
	double* v = (double*) calloc(ncol,sizeof(double));
	this->getJacobi(v);
	this->diag_x_sparse(v);
	return v;
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
            std::cout << setw(8) << matrix[i][j] << " ";
        }
        std::cout << endl;
    }

    // Free the memory allocated for the matrix
    for (int i = 0; i < nrow; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void sparse::printComponents(){
	std::cout<<"Coef: ";
	for (int i=0; i<n_term; i++){std::cout<<coef[i]<<" ";}
	std::cout<<std::endl<<"ja: ";
	for (int i=0; i<n_term; i++){std::cout<<ja[i]<<" ";}
	std::cout<<std::endl<<"iat: ";
	for (int i=0; i<nrow+1; i++){std::cout<<iat[i]<<" ";}
	std::cout<<std::endl;
}

#endif