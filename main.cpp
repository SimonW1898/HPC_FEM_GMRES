#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include "sparse.h"
#include "read_mat.h"
#include "functions.h"
#include "gmres.h"

const int n_threads = 8;
using namespace std;

int main(){
	// char name[] = "Cubo_591.csr";
	// ifstream Cubo_591;
	// Cubo_591.open(name);

	// int n, n, nt;
	// Cubo_591 >> n >> n >> nt;





	// // Read and Save a Matrix to Multiply and test things
	// char name_multi[] = "multi.rig";
	// ifstream multi;
	// multi.open(name_multi);

	// int nr_mult, nc_mult, nt_mult;
	// multi >> nr_mult >> nc_mult >> nt_mult;
	// int *iat_mult,*ja_mult;

	// double *elem_mult;

	// sparse B(nr_mult, nc_mult ,nt_mult);
	// readCSRmat(name_multi,&nr_mult, &nc_mult, &nt_mult,&iat_mult,&ja_mult,&elem_mult,false);
	// B.set_iat(iat_mult);
	// B.set_ja(ja_mult);
	// B.set_coef(elem_mult);


	char fname[] = "data_input/Cubo_591.rig";
	
	sparse mat = sparse();
	createMat(&mat,fname);


	// create some vectors to test things
	double *b,*y;
	int n = mat.get_ncol();

	b = (double*)malloc((n)*sizeof(double));
	y = (double*)malloc((n)*sizeof(double));


	for (int i = 0; i < n; ++i){
		b[i] = i + 1;
		y[i] = n - i;
	}

	
	/*Matrix Vector Prodct*/
	//allocate memory for the solution of the Matrix Vector prod

	double *xstar = (double*)malloc((n)*sizeof(double));

	gmres(&mat,b, xstar);

	free(b);
	free(y);
	free(xstar);


// scalar signum function
	// cout<<scalarsgn(3.0);


// Generate random values between 0 and 1
    // for (int i = 0; i < n; ++i) {
    //     v[i] = static_cast<double>(rand()) / RAND_MAX;
    // }


// post pre MV multiplication
	// mat.post_MV(x,b); 
	// mat.pre_MV(x,v);
	// mat.printMtx();
	// printVec(v,n);
	// printVec(x, mat.get_ncol());
	// printVec(b, mat.get_nrow());
	// printVec(v, mat.get_ncol());


// Normalization
	// cout << norm(v, n) << endl;
	// normalize(v,n);
	// printVec(v, mat.get_ncol());
	// cout << norm(v, n) << endl;
	// normalize_copy(c,b,n);
	// printVec(c,n);
	// cout << norm(c, n) << endl;

    
/* Try evaluate the run time */

	// int repetitions = 100;
    // double start_time =	 omp_get_wtime();
    // for(int i=0;i<repetitions;i++){
    // 	mat.matrixVectorProduct(v,b); 
    // 	scalar_prod(v,v,n);
    // 	vector_update(v,v,-1.0,n);
    // }
	
 	// double execution_time = (omp_get_wtime()-start_time);

    // cout << "Execution Time: " << execution_time << " seconds" << std::endl;
    


    /* Matrix Multiplication */

	// sparse sol(n,nc_mult,n*nc_mult);

    // sparse* sol_ptr = &sol;
    // sparse* B_ptr = &B;

    // mat.matrixProduct(B_ptr,sol_ptr);
    // sol.printMtx();
    // printVec(sol.get_iat(), mat.get_nrow()+1);
	// printVec(mat.get_ja(), mat.get_n_term());




    return 0;
}
