// #ifndef GMRES_H
// #define GMRES_H

#include <iostream>
#include "sparse.h"
#include "constants.h"
#include "functions.h"
#include <cmath>
#include <omp.h>


void gmres(sparse* A_ptr, double* b, double* x){
	/*
	GMRES Algorithm with
		- Householder projections
		- Givens rotation

	Input:
		- A_ptr: pointer to sparse Matrix of the linear system to be solved
		- b: ptr to rhs
		- x: ptr to solution vector (and start vector?)
	




	Flag is set to see the status of the Algorithm
	flag = 0: succesful convergence
		- inner loop: rormr_act < tolb
		- outer loop: rormr_act < tolb after the last evaluation
	flag = 1: maximum iterations reached without convergence
		- initialization if it stays unchanged: max iterations reached
	flag = 2: stagnation: no change after few iterations
		- used only with preconditioners
	flag = 3: error 
		- moresteps >= maxmsteps: tolerance to large
		- stag>=maxstagsteps
	*/

	sparse& A = *A_ptr;
	int nc = A.get_ncol();
	int nr = A.get_nrow();

	if(nc!=nr){
		cout<<"ERROR need square Matrix n != n"<<endl;
		return;
	}

	const int n = nc;
	

	double tol = 1e-14;
	int maxit = n; // 10 is small
	double n2b = norm(b,n);
	double* minv_b = (double*)malloc((n)*sizeof(double));
	vector_update(minv_b,b,1,n); // copying
	int inner = maxit;
	int outer = maxit;

	int flag = 1;
	double* xmin = (double*)malloc((n)*sizeof(double));
	vector_update(xmin,b,1,n); //copying
	int imin = 0;
	int jmin = 0;
	double tolb = tol*n2b;
	int evalxm = 0;
	int stag = 0;
	int moresteps = 0;
	int maxmsteps = 3;
	int maxstagsteps = 3;
	int minupdated = 0;


	double* r = (double*)malloc((n)*sizeof(double));
	double* Ax = (double*)malloc((n)*sizeof(double));
	double* xm = (double*)malloc((n)*sizeof(double));
	double* Axm = (double*)malloc((n)*sizeof(double));

	A.post_MV(x,Ax);
	vector_add(1.,b,-1.,Ax,r,n); 
	double normr = norm(r, n);
	double n2minv_b = norm(minv_b, n);

	double* resvec = (double*)malloc((inner*outer+1)*sizeof(double));
	resvec[0] = normr;
	double normrmin = normr;

	// preallocate
	double** J = (double**)malloc(2 * sizeof(double*));	// Matrix for givens rotation J(2,inner)
    for (int i = 0; i < inner; ++i) {
        J[i] = (double*)malloc(inner * sizeof(double));
    }

	double** U = (double**)malloc(n * sizeof(double*));	// Matrix for holding Householder reflectors u(w) U(n,inner)
    for (int i = 0; i < inner; ++i) {
        U[i] = (double*)malloc(inner * sizeof(double));
    }
    double* u = (double*)malloc(n * sizeof(double)); // row vector of u as helper;

    double** R = (double**)malloc(inner * sizeof(double*));	// givens rotated Hessenbergmatrix R(inner,inner) 
    for (int i = 0; i < inner; ++i) {
        R[i] = (double*)malloc(inner * sizeof(double));
    }

    double* w = (double*)malloc((inner+1)*sizeof(double));

    double* v = (double*)malloc((inner+1)*sizeof(double));
    double* vtemp = (double*)malloc((inner+1)*sizeof(double));

	double* hhhh = (double*)malloc(2*sizeof(double));
    double* jtemp = (double*)malloc(2*sizeof(double)); // there to save v(initer:initer+1)


	free(hhhh);
    double beta;
    double alpha;

    double normr_act;

    double* minv_r = (double*)malloc((inner+1)*sizeof(double));

    double relres; //relative residual

    int outiter, initer; // so that they live outside the loop

    int idx;


    for(outiter = 0; outiter<outer; outiter++){
    	normr = norm(r,n);
    	beta = scalarsgn(normr)*normr;
    	copyArray(u, r, n);
    	u[0] += beta;
    	normalize(u, n);
    	copyArr2Mat_row(U,u,0,n);
    	//  Apply Householder projection to r.
    	//  w = r - 2*u*u'*r;
    	w[0] = -beta;

		
    	for(initer=0; initer<inner; initer++){
    		copyArray(v, u, n);
    		vector_scalarMult(v, u, -2.0*u[initer], n);
    		v[initer] += 1;

    		// v = P1*P2*...Pjm1*(Pj*ej)
    		for(int k = initer - 1; k >= 0; k--){
				copyMat_row2Arr(U, u, k, n);
				vector_update(v, u, -2*scalar_prod(u,v,n), n);
			}
			normalize(v, n);
			copyArray(vtemp, v, n);

			A.post_MV(vtemp, v);
			
			// Form Pj*Pj-1*...*P1*Av
			for(int k=0; k<=initer; k++){
				copyMat_row2Arr(U, u, k, n);
				vector_update(v, u, -2*scalar_prod(u,v,n), n);
			}


			// determine Pj+1
			if(initer!= n-1){ // if not last iteration
				// construct u for Pj+1
				copyArray(u, v, n);
				vector_zero_out(u, 0, initer, n);

				alpha = norm(u,n);

				if (alpha > eps){
					alpha *= scalarsgn(v[initer+1]);
					//u = v(initer+1:end) + sign(v(initer+1))*||v(initer+1:end)||*e_{initer+1)
					u[initer+1] += alpha;

					normalize(u, n);

					copyArr2Mat_row(U, u, initer+1, n);

					// Apply Pj+1 to v
					vector_zero_out(v, initer+1, n-1, n);
					v[initer+1] -= alpha;
				}
			}


			for(int colJ = 0; colJ<initer; colJ++){
				double tmpv = v[colJ];
				v[colJ] = J[0][colJ]*v[colJ] + J[1][colJ]*v[colJ+1];
				v[colJ+1] = -J[1][colJ]*tmpv + J[0][colJ]*v[colJ+1];
			}


			// compute Given's rotation Jm
			if (initer!= n-1){
				jtemp[0] = v[initer];
				jtemp[1] = v[initer+1];

				double rho = norm(jtemp,2);
				normalize(jtemp,2);
				copyArr2Mat_row(J,jtemp,initer,2);

				w[initer+1] = -J[1][initer]*w[initer];
				w[initer] *= J[0][initer];

				v[initer] = rho;
				v[initer+1] = 0;
			}


			copyArr2Mat_row(R,v,initer,inner);

			normr = scalarsgn(w[initer+1])*w[initer+1];		// abs(w[initer+1])
			resvec[outiter*inner+initer+1] = normr;
			normr_act = normr;


			if (normr <= tolb || stag >= maxstagsteps || moresteps){ 
			// normr smaller than relative tolerance
			// stagnation larger than allowed
			// if moresteps set to 1 to allow further improvement 
			// one of these is true: allow to compute the solution vector x

				double* additive = (double*)malloc(n*sizeof(double));
				double* ytemp = (double*)malloc((initer+1)*sizeof(double));

				triangularSolver(R,w,ytemp,initer+1);
				copyMat_row2Arr(U, u, initer, n);
				vector_scalarMult(additive,u,-2*ytemp[initer]*U[initer][initer],n);
				additive[initer] += ytemp[initer];
				for(int k=initer-1;k>=0;k--){
					copyMat_row2Arr(U, u, k, n);
					additive[k] += ytemp[k];
					double tmpscalar = -2*scalar_prod(u,additive,n);
					vector_update(additive,u,tmpscalar,n);
				}
				if(norm(additive,n)<eps*norm(x,n)){ //check if the additive vector is big enough to have an influence
					stag += 1;
				}else{
					stag = 0;
				}

				copyArray(xm,x,n);
				vector_update(xm,additive,1,n);	// xm = x + additive
				evalxm = 1;
				free(ytemp);
				
				free(additive);



				A.post_MV(xm,Axm);
				vector_add(1.,b,-1.,Axm,r,n); 

				if (norm(r,n)<= tolb){
					copyArray(x,xm,n);
					flag = 0;
					imin = outiter;
					jmin = initer;
					break;
				}
				copyArray(minv_r,r,n);
				normr_act = norm(minv_r,n);
				resvec[outiter*inner+inner+1] = normr_act;


				// space for preconditioner

				if (normr_act <= normrmin){
					normrmin = normr_act;
					imin = outiter;
					jmin = initer;
					copyArray(xmin,xm,n);
					minupdated = 1;
				}

				if (normr_act <= tolb){
					copyArray(x,xm,n);
					flag = 0;
					imin = outiter;
					jmin = initer;
					break;
				}else{
					if (stag >= maxstagsteps && moresteps ==0){
						stag = 0;
					}
					moresteps += 1;
					if (moresteps >= maxmsteps){
						cout<<"tolerance to small"<<endl;
						flag = 3;
						imin = outiter;
						jmin = initer;
						break;
					}
				}
    		}//endif (normr <= tolb || stag >= maxstagsteps || moresteps)

			if(normr_act<=normrmin){
				normrmin = normr_act;
				imin = outiter;
				jmin = initer;
				minupdated = 1;
			}

			if (stag>=maxstagsteps){
				flag = 3;
				break;
			}
		} // end for in

		evalxm =0;
		if (flag != 0){	// computes the x with the lowest norm
			if (minupdated){ // if the minimal norm has been updated 
				idx = jmin;
			}else{
				idx = initer;
			}	
			if (idx >= 0){ // some iterations were performed
				double* additive = (double*)malloc(n*sizeof(double));
				double* y = (double*)malloc((initer+1)*sizeof(double));
				triangularSolver(R,w,y,initer+1);
				copyMat_row2Arr(U, u, initer, n);
				vector_scalarMult(additive,u,-2*y[initer]*U[initer][initer],n);
				additive[initer] += y[initer];
				for(int k=initer-1;k>=0;k--){
					copyMat_row2Arr(U, u, k, n);
					additive[k] += y[k];
					double tmpscalar = -2*scalar_prod(u,additive,n);
					vector_update(additive,u,tmpscalar,n);
				}
				vector_update(x,additive,1,n);
				free(additive);
				free(y);
			}

			copyArray(xmin,x,n);
			A.post_MV(x,Ax);
			vector_add(1.0,b,-1.0,Ax,r,n);
			copyArray(minv_r,r,n);

			normr_act = norm(minv_r,n);

			if (inner%20==1){
				int iterations = (initer+1)*(outiter+1);
				// writeCSV(n, inner, iterations+1, x, w, resvec, U, R);
				cout<<"GMRES iteration, with residual norm: " << normrmin<< endl;
			}
		}// end if flag~=0

		if (normr_act<normrmin){
			copyArray(xmin,x,n);
			normrmin = normr_act;
			imin = outiter;
			jmin = initer;
		}

		if (flag == 3){
			break;
		}

		if (normr_act <= tolb){
			flag = 0;
			break;
		}
		minupdated = 0;
    }//end for out

    if (initer == 0 && outiter ==0){
    	normr_act = normrmin;
    }

    if (flag == 0){
    	relres = normr_act/n2minv_b;	
    }else{
    	copyArray(x,xmin,n);
    	relres = normr_act/n2minv_b;	
    }

	
    int iterations = (initer+1)*(outiter+1);


    writeCSV(n, inner, iterations+1,b, x, w, resvec, U, R);
    cout<<"GMRES finished, with residual norm: " << normrmin<< endl << "flag: "<< flag<< " initer: " << initer <<" outiter: "<< outiter<< endl;

	free(minv_b);
	free(xmin);
	free(r);
	free(Ax);
	free(xm);
	free(Axm);

	for (int i = 0; i < 2; ++i) {
        free(J[i]);
    }
    free(J);

	free(resvec);

	for (int i = 0; i < n; ++i) {
        free(U[i]);
    }
    free(U);

	free(u);

    for (int i = 0; i < inner; ++i) {
        free(R[i]);
    }
    free(R);

    free(w);

    free(v);

    free(vtemp);

    free(jtemp);

	free(minv_r);

}// end for out this is end of gmres



    
    
 






// #endif