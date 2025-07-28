void gmres(char* matname, sparse* A_ptr, double* b, double* x, bool preconditioning, bool restarting, int maxit){
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
	flag = 2: stagnation 
		- no significant change in norm reduction

	*/
	sparse& A = *A_ptr;

	int nc = A.get_ncol();
	int nr = A.get_nrow();

	if(nc!=nr){
		cout<<"ERROR need square Matrix n != n"<<endl;
		return;
	}

	const int n = nc;

	double make_relres_smaller=1;
	if (preconditioning){
		double* M_inv = A.left_Jacobi();
		elementwise_prod(b,M_inv,n);
	}else{
		make_relres_smaller = 1.0e-15;
	}

	int restart, outer;
	if(restarting){
		restart = 20;
		outer = maxit/restart;
		maxit = restart; // 10 is small
	}
	int inner = maxit;




	double tol = 1e-15;
	double n2b = norm(b,n);


	int flag = 1;

	int imin = 0;
	int jmin = 0;
	double tolb = tol*n2b*make_relres_smaller;

	int stag = 0;
	int moresteps = 0;
	int maxmsteps = 3;
	int maxstagsteps = 3;


	double* r = (double*)malloc((n)*sizeof(double));
	double* Ax = (double*)malloc((n)*sizeof(double));
	double* xm = (double*)malloc((n)*sizeof(double));
	double* Axm = (double*)malloc((n)*sizeof(double));

	A.post_MV(x,Ax);
	vector_add(1.,b,-1.,Ax,r,n); 
	double normr = norm(r, n);

	double* resvec = (double*)malloc((inner*outer+1)*sizeof(double));
	resvec[0] = normr;
	double normrmin = normr;

	// preallocate

	// J is of size (inner x 2)
	// Values used in one iteration are in the same array
	// working direction J[row][i] ist ja doch die falsche working direction oder
	double** J = (double**)malloc(inner * sizeof(double*));	// Matrix for givens rotation J(2,inner)
    for (int i = 0; i < inner; ++i) {
        J[i] = (double*)malloc(2 * sizeof(double));
    }

	// U is of size (inner x n)
	// Values used in one iteration are in the same array
	// working direction U[row][i] = i;
	double** U = (double**)malloc((inner+1) * sizeof(double*));	// Matrix for holding Householder reflectors u(w) U(n,inner)
    for (int i = 0; i < (inner+1); ++i) {
        U[i] = (double*)malloc(n * sizeof(double));
    }


    double* u = (double*)malloc(n * sizeof(double)); // row vector of u as helper;

    double** R = (double**)malloc(inner * sizeof(double*));	// givens rotated Hessenbergmatrix R(inner,inner) 
    for (int i = 0; i < inner; ++i) {
        R[i] = (double*)malloc(inner * sizeof(double));
    }

    double* w = (double*)malloc((n+1)*sizeof(double));

    double* v = (double*)malloc((n+1)*sizeof(double));
    double* vtemp = (double*)malloc((n+1)*sizeof(double));

    double* jtemp = (double*)malloc(2*sizeof(double)); // there to save v(initer:initer+1)


    double beta;
    double alpha;


    double relres; //relative residual

    int initer; // so that they live outside the loop
	int outiter;
	int iterations = 0;
    int idx;

	double* y = (double*)malloc((inner+1)*sizeof(double));
	double* additive = (double*)malloc(n*sizeof(double));


	double start_time = omp_get_wtime();

	normr = norm(r,n);
	beta = scalarsgn(normr)*normr;
	copyArray(u, r, n);
	u[0] += beta;
	normalize(u, n);

	copyArr2Mat_col(U,u,0,n);
	//  Apply Householder projection to r.
	//  w = r - 2*u*u'*r;
	w[0] = -beta;

	for (outiter = 0; outiter < outer; outiter++){
		for(initer=0; initer<inner; initer++){
				iterations = (outiter)*inner+initer+1;
				// copyArray(v, u, n);		// wofuer das
				vector_scalarMult(v, u, -2.0*u[initer], n);
				v[initer] += 1;

				// v = P1*P2*...Pjm1*(Pj*ej)
				for(int k = initer - 1; k >= 0; k--){
					// copyMat_col2Arr(U, u, k, n);
					// vector_update(v, u, -2*scalar_prod(u,v,n), n);
					vector_update(v, U[k], -2*scalar_prod(U[k],v,n), n);

				}

				normalize(v, n);
				copyArray(vtemp, v, n);

				A.post_MV(vtemp, v);
				
				// Form Pj*Pj-1*...*P1*Av
				for(int k=0; k<=initer; k++){
					// copyMat_col2Arr(U, u, k, n);
					// vector_update(v, u, -2*scalar_prod(u,v,n), n);
					vector_update(v, U[k], -2*scalar_prod(U[k],v,n), n);

				}


				// determine Pj+1
				if(initer!= n-1){ // if not last iteration // not necessary in restarted version
					// construct u for Pj+1
					copyArray(u, v, n);
					vector_zero_out(u, 0, initer, n);

					alpha = norm(u,n);

					if (alpha > eps){
						alpha *= scalarsgn(v[initer+1]);
						//u = v(initer+1:end) + sign(v(initer+1))*||v(initer+1:end)||*e_{initer+1)
						u[initer+1] += alpha;

						normalize(u, n);

						copyArr2Mat_col(U, u, initer+1, n);

						// Apply Pj+1 to v
						vector_zero_out(v, initer+1, n-1, n);
						v[initer+1] -= alpha;
					}
				}


				for(int colJ = 0; colJ<initer; colJ++){
					double tmpv = v[colJ];
					v[colJ] = J[colJ][0]*v[colJ] + J[colJ][1]*v[colJ+1];
					v[colJ+1] = -J[colJ][1]*tmpv + J[colJ][0]*v[colJ+1];
				}


				// compute Given's rotation Jm
				if (initer!= n-1){
					jtemp[0] = v[initer];
					jtemp[1] = v[initer+1];

					double rho = norm(jtemp,2);
					normalize(jtemp,2);
					copyArr2Mat_col(J,jtemp,initer,2);

					w[initer+1] = -J[initer][1]*w[initer];
					w[initer] *= J[initer][0];

					v[initer] = rho;
					v[initer+1] = 0;
				}


				copyArr2Mat_row(R,v,initer,inner);

				normr = scalarsgn(w[initer+1])*w[initer+1];		// abs(w[initer+1])
				resvec[iterations] = normr;


				if (normr <= tolb || stag >= maxstagsteps || moresteps){ 
				// normr smaller than relative tolerance
				// stagnation larger than allowed
				// if moresteps set to 1 to allow further improvement 
				// one of these is true: allow to compute the solution vector x

					vector_zero_out(additive,n);
					vector_zero_out(y,initer+1);

					triangularSolver(R,w,y,initer+1);
					copyMat_col2Arr(U, u, initer, n);
					vector_scalarMult(additive,u,-2*y[initer]*U[initer][initer],n);
					additive[initer] += y[initer];
					for(int k=initer-1;k>=0;k--){
						// copyMat_col2Arr(U, u, k, n);
						// additive[k] += y[k];
						// double tmpscalar = -2*scalar_prod(u,additive,n);
						// vector_update(additive,u,tmpscalar,n);

						additive[k] += y[k];
						double tmpscalar = -2*scalar_prod(U[k],additive,n);
						vector_update(additive,U[k],tmpscalar,n);
					}
					if(norm(additive,n)<eps*normr){ //check if the additive vector is big enough to have an influence
						stag += 1;
					}else{
						stag = 0;
					}

					// copyArray(xm,x,n);
					// vector_update(xm,additive,1,n);	// xm = x + additive
					vector_add(1,x,1,additive,xm,n);
					



					A.post_MV(xm,Axm);
					vector_add(1.,b,-1.,Axm,r,n); 
					if (norm(r,n)<= tolb){
						copyArray(x,xm,n);
						flag = 0;
						break;
					}
					normr = norm(r,n);

					resvec[iterations] = normr;





					if (normr <= tolb){
						copyArray(x,xm,n);
						flag = 0;
						jmin = initer;
						break;
					}else{
						if (stag >= maxstagsteps && moresteps ==0){
							stag = 0;
						}
						moresteps += 1;
						if (moresteps >= maxmsteps){
							cout<<"stagnation"<<endl;
							flag = 2;
							jmin = initer;
							break;
						}
					}
				}//endif (normr <= tolb || stag >= maxstagsteps || moresteps)


				if (stag>=maxstagsteps){
					flag = 2;
					break;
				}
		}// end for in

		if (flag != 0){	// computes the x with the lowest norm				
			vector_zero_out(additive,n);

			vector_zero_out(y,initer+1);
			triangularSolver(R,w,y,initer);
			// copyMat_col2Arr(U, u, initer, n);
			// vector_scalarMult(additive,u,-2*y[initer]*U[initer][initer],n);

			vector_scalarMult(additive,U[initer],-2*y[initer]*U[initer][initer],n);

			
			additive[initer] += y[initer];
			for(int k=initer-1;k>=0;k--){
				// copyMat_col2Arr(U, u, k, n);
				// additive[k] += y[k];
				// double tmpscalar = -2*scalar_prod(u,additive,n);
				// vector_update(additive,u,tmpscalar,n);

				additive[k] += y[k];
				double tmpscalar = -2*scalar_prod(U[k],additive,n);
				vector_update(additive,U[k],tmpscalar,n);
			}

			vector_update(x,additive,1,n);

			A.post_MV(x,Ax);
			vector_add(1.0,b,-1.0,Ax,r,n);

			normr = norm(r,n);

			normr = norm(r,n);
			beta = scalarsgn(normr)*normr;
			
			vector_zero_out(u,inner);
			mat_zero_out(U,n,inner+1);
			copyArray(u, r, n);
			u[0] += beta;
			normalize(u, n);

			copyArr2Mat_col(U,u,0,n);
			//  Apply Householder projection to r.
			//  w = r - 2*u*u'*r;
			w[0] = -beta;
		}// end if flag~=0


		if (flag == 2) { //stagnation
			break;
		}

		if (normr <= tolb) { // convergence
			flag = 0;
			break;
		}


	} //end for out




	if (normr <= tolb){
		flag = 0;
	}


	double end_time = omp_get_wtime();
    double runtime = (end_time - start_time);

    // writeCSV(matname,n, inner, iterations,b, x, w, resvec, U, R);
    cout<<"GMRES finished, with residual norm: " << normr<< endl << "flag: "<< flag<< " iterations: " << iterations << endl;
    // cout << "Runtime: " << runtime << " seconds" << endl;

	free(r);
	free(Ax);
	free(xm);
	free(Axm);

	for (int i = 0; i < 2; ++i) {
        free(J[i]);
    }
    free(J);

	free(resvec);

	for (int i = 0; i < inner+1; ++i) {
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

	free(y);
	free(additive);

}//this is end of gmres
