#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <cstring>

#include "sparse.h"
#include "read_mat.h"
#include "functions.h"
#include "GMRES.h"
#include "constants.h"
#include "fem_functions.h"


const int n_threads = 8;
using namespace std;

int main(int argc, char* argv[]) {
	//allocate memory for the solution of the Matrix Vector prod
    // char filename[] = "mesh/u_mattia.dat";

	// sparse mat_mattia = sparse();
	// double* b_mattia = getFEM_sparse(&mat_mattia);

	// double *xstar_mattia = (double*)malloc(mat_mattia.get_nrow()*sizeof(double));

	// // sparse* newmat = &mat_mattia;
	// gmres(filename, &mat_mattia,b_mattia, xstar_mattia, true, true, 1000);

    // save_solution(filename, xstar_mattia, mat_mattia.get_ncol());

	// free(b_mattia);
	// free(xstar_mattia);




	// FEM assembly

	


	//     int num_threads = 4;
    // omp_set_num_threads(num_threads);

    // Enable dynamic threads to support task-based parallelism
    // omp_set_dynamic(1);

    // Get the actual number of threads being used by OpenMP
    int actual_num_threads = omp_get_max_threads();
    std::cout << "Actual number of threads: " << actual_num_threads << std::endl;


	double start_time = omp_get_wtime();

	double D[2] = {1,1};
	double V[2] = {1,3};
	int ne, nn, nb; // number of mesh (800), number of nodes(441), number of ? (80)

	double **coord, **bound, *delta, *delta_node, *F;
	double **topol, *temp, *q, *b;



	sparse* FEM;
	bool bigmesh = true;

	char sol_path[100];
	
	
	if (bigmesh){
		string mesh_path = "bigmesh/mesh.dat";
		string nodes_path = "bigmesh/dirtot.dat";
		string coord_path = "bigmesh/xy.dat";

		ifstream mesh(mesh_path);
		ifstream nodes(nodes_path);
		ifstream xy(coord_path);


		xy >> nn;
		nodes >> nb;
		mesh >> ne;

		bound = mat_allocation(nb,2);
		coord = mat_allocation(nn,2);
		topol = mat_allocation(ne,3);

		for (int i = 0; i < nn; ++i){xy >> coord[i][0] >> coord[i][1];}
		for (int i = 0; i < nb; ++i){nodes >> bound[i][0] >> bound[i][1];}
		for (int i = 0; i < ne; ++i){mesh >> topol[i][0] >> topol[i][1] >> topol[i][2];}

		xy.close();
		nodes.close();
		mesh.close();
		char path[] = "bigmesh/u.dat";
		strcpy(sol_path, path);


	}else{
		string mesh_path = "mesh/mesh.dat";
		string nodes_path = "mesh/dirnod.dat";
		string coord_path = "mesh/xy.dat";


		ifstream mesh(mesh_path);
		ifstream nodes(nodes_path);
		ifstream xy(coord_path);

		int not_used;

		xy >> nn;
		nodes >> nb;
		mesh >> ne;


		bound = mat_allocation(nb,2);
		coord = mat_allocation(nn,2);
		topol = mat_allocation(ne,3);

		for (int i = 0; i < nn; ++i){xy >> coord[i][0] >> coord[i][1];}
		for (int i = 0; i < nb; ++i){nodes >> bound[i][0] >> bound[i][1];}
		for (int i = 0; i < ne; ++i){mesh >> topol[i][0] >> topol[i][1] >> topol[i][2]>>not_used;}


		for (int i = 0; i < nn; ++i){xy >> coord[i][0] >> coord[i][1];}
		for (int i = 0; i < nb; ++i){nodes >> bound[i][0] >> bound[i][1];}
		for (int i = 0; i < ne; ++i){mesh >> topol[i][0] >> topol[i][1] >> topol[i][2];}

		xy.close();
		nodes.close();
		mesh.close();
		char path[] = "mesh/u.dat";
		strcpy(sol_path, path);
	}


	//read data



	// printMat(coord,nb,2);

	

	double h = sqrt(2)*(coord[0][1]-coord[1][1]);// why sqrt2
	double tau = 0.1; // whats tau?

	delta = (double*)malloc(ne*sizeof(double));

	delta_node = (double*)calloc(nn,sizeof(double));

	F = (double*)malloc(nn*sizeof(double));
	q = (double*)calloc(nn,sizeof(double));
	b = (double*)malloc(nn*sizeof(double));


	FEM = create_FEM_mat(coord, topol, ne, nn, D, h, tau, V, delta);

	int idx;
    #pragma omp for schedule(dynamic,nn/1000+1)
    for(int i = 0; i<ne; i++){
        for(int k=0; k<3; k++){
            idx = static_cast<int>(floor(topol[i][k])-1);
			delta_node[idx] += delta[k]/3;
		}
	}

	// RHS
	for(int i = 0; i<nn; i++){
		F[i] = force(coord[i][0],coord[i][1]*delta_node[i]);
	}

	// BCs
	imposeBC(&FEM[0], bound, q, nb);
	FEM[1].addition_update(&FEM[2]);
	FEM[0].addition_update(&FEM[1]);
	// printVec(FEM[0].get_coef(),FEM[0].get_n_term());
	double* x = (double*)calloc(FEM[0].get_ncol(),sizeof(double));

	
	vector_update(q,F,1,nn);
	// printVec(q,nn);

	// FEM Solver
	cout<< norm(q,nn)<<endl;
	sparse* A = new sparse;
	// b = getAssembledSystem(A);

	// double* x = (double*)calloc(A->get_ncol(),sizeof(double));
	// FEM[0].printComponents();

	
	gmres(sol_path,&FEM[0],q, x, true, true, 1000);
	double end_time = omp_get_wtime();
    double runtime = (end_time - start_time);
	
	save_solution(sol_path,x,FEM[0].get_ncol());


	free(b);
	free(x);
	delete A;

    return 0;

    }