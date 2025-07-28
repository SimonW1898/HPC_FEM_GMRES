#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include "constants.h"
#include "sparse.h"
#include "functions.h"

#include <fstream>
#include <string>
#include <cmath> 

#include <algorithm>
#include <vector>

using namespace std;

void stiffness_struct(char* fname, int ne, int nn, double** topol) {
    double** Adj = mat_allocation(ne, nn);
    double* i_idx = (double*)malloc(static_cast<int>(nn*nn*0.2));
    double* j_idx = (double*)malloc(static_cast<int>(nn*nn*0.2));

    int idx;
    #pragma omp parallel for schedule(dynamic, nn/1000+1)
    for (int i = 0; i < ne; i++) {
        for (int k = 0; k < 3; k++) {
            idx = static_cast<int>(floor(topol[i][k]) - 1);
            Adj[i][idx] = 1;
        }
    }

    double sum;
    int nnz = 0;
    std::string tempFile = "temp_file.txt";
    std::ofstream file(tempFile, std::ofstream::trunc); // Open the temporary file in write mode with truncation
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            sum = 0;
            for (int k = 0; k < ne; k++) {
                sum += Adj[k][i] * Adj[k][j];
            }
            if (sum > 0) {
                nnz++;
                file << j + 1 << " " << i + 1 << " " << 0 << std::endl;
            }
        }
    }
    file.close();

    // Reopen the temporary file in read mode
    std::ifstream inFile(tempFile);
    if (!inFile) {
        std::cerr << "Error opening the temporary file for reading." << std::endl;
        return;
    }

    // Open the original file in write mode with truncation
    std::ofstream outFile(fname, std::ofstream::trunc);
    if (!outFile) {
        std::cerr << "Error opening the file for writing." << std::endl;
        inFile.close();
        return;
    }

    // Write the new data at the beginning of the original file
    outFile << nn << " " << nn << " " << nnz << std::endl;

    // Copy the rest of the content from the temporary file to the original file
    std::string line;
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            outFile << line << std::endl;
        }
    }

    // Close both files
    inFile.close();
    outFile.close();

    // Delete the temporary file
    std::remove(tempFile.c_str());
}


void stiffness_struct_para(char* fname, int ne, int nn, double** topol) {
    double** Adj = mat_allocation(ne, nn);
    double* i_idx = (double*)malloc(static_cast<int>(nn*nn*0.2));
    double* j_idx = (double*)malloc(static_cast<int>(nn*nn*0.2));

    int idx;
    #pragma omp parallel for schedule(dynamic, nn/1000+1)
    for (int i = 0; i < ne; i++) {
        for (int k = 0; k < 3; k++) {
            idx = static_cast<int>(floor(topol[i][k]) - 1);
            Adj[i][idx] = 1;
        }
    }

    double sum;
    int nnz = 0;
    std::string tempFile = "temp_file.txt";
    std::ofstream file(tempFile, std::ofstream::trunc); // Open the temporary file in write mode with truncation
    
    #pragma omp for schedule(dynamic,nn/1000+1)
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            sum = 0;
            for (int k = 0; k < ne; k++) {
                if (Adj[k][i] * Adj[k][j] > 0) {
                nnz++;
                file << j + 1 << " " << i + 1 << " " << 0 << std::endl;
                break;
            }
            }
            
        }
    }
    file.close();

    // Reopen the temporary file in read mode
    std::ifstream inFile(tempFile);
    if (!inFile) {
        std::cerr << "Error opening the temporary file for reading." << std::endl;
        return;
    }

    // Open the original file in write mode with truncation
    std::ofstream outFile(fname, std::ofstream::trunc);
    if (!outFile) {
        std::cerr << "Error opening the file for writing." << std::endl;
        inFile.close();
        return;
    }

    // Write the new data at the beginning of the original file
    outFile << nn << " " << nn << " " << nnz << std::endl;

    // Copy the rest of the content from the temporary file to the original file
    std::string line;
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            outFile << line << std::endl;
        }
    }

    // Close both files
    inFile.close();
    outFile.close();

    // Delete the temporary file
    std::remove(tempFile.c_str());
}


void stiffness_struct_perf(sparse* H, int ne, int nn, double** topol){
    double** Adj = mat_allocation(ne+1, nn);

    // Create vectors for i_idx and j_idx
    std::vector<int> i_idx;
    std::vector<int> j_idx;
    int idx;

    // Reserve an initial capacity for the vectors 
    double n_temp = static_cast<double>(nn);
    int initialCapacity = static_cast<int>(n_temp * n_temp * 0.05 );
    i_idx.reserve(initialCapacity);
    j_idx.reserve(initialCapacity);

    #pragma omp parallel for schedule(dynamic, nn/1000+1)
    for (int i = 0; i < ne; i++) {
        for (int k = 0; k < 3; k++) {
            idx = static_cast<int>(floor(topol[i][k]) - 1);
            Adj[i][idx] = 1;
        }
    }

    double sum;
    int nnz = 0;
    #pragma omp parallel for schedule(dynamic,nn/1000+1)
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            sum = 0;
            for (int k = 0; k < ne; k++) {
                if (Adj[k][i] * Adj[k][j] > 0) {
                    j_idx[nnz] = (i +1);
                    i_idx[nnz] = (j);
                    #pragma omp atomic
                    nnz++;
                    // Optionally, check the capacity and reserve more space if needed (e.g., double the current capacity)
                    if (nnz >= i_idx.capacity()) {
                        std::cout<<"capacity to small"<<nnz<<endl;
                        int newCapacity = i_idx.capacity() * 1.5;
                        i_idx.reserve(newCapacity);
                        j_idx.reserve(newCapacity);
                    }

                    break;
                }
            }
        }
    }
    int* ja_loc = &i_idx[0];
    int* irow = &j_idx[0];

    int* iat_loc = (int*)malloc(nn*sizeof(int));
    irow2iat(nn,nnz,irow,iat_loc);
    H->set_n_term(nnz);
    H->set_ncol(nn);
    H->copy_ja(ja_loc);
    H->copy_iat(iat_loc);

    free(iat_loc);
}

void stiffness_struct_small_loops(sparse* H, int ne, int nn, double** topol) {
       // Create vectors to store the i_idx and j_idx pairs
    std::vector<int> i_idx;
    std::vector<int> j_idx;

    // Loop through each element
    for (int i = 0; i < ne; i++) {
        for (int k = 0; k < 3; k++) {
            int idx = static_cast<int>(floor(topol[i][k]) - 1);
            for (int l = 0; l < 3; l++) {
                int jdx = static_cast<int>(floor(topol[i][l]) - 1);
                i_idx.push_back(idx);
                j_idx.push_back(jdx);
            }
        }
    }

    // Combine i_idx and j_idx into pairs for sorting
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < i_idx.size(); i++) {
        pairs.push_back(std::make_pair(i_idx[i], j_idx[i]));
    }

    // Sort the pairs to bring duplicates together
    std::sort(pairs.begin(), pairs.end());

    // Remove duplicates and copy the unique pairs back to i_idx and j_idx
    int nnz = 0;
    for (int i = 0; i < pairs.size(); i++) {
        if (i == 0 || pairs[i] != pairs[i - 1]) {
            i_idx[nnz] = pairs[i].first + 1;
            j_idx[nnz] = pairs[i].second;
            nnz++;
        }
    }

    // Allocate memory for the CSR arrays
    int* ja_loc = new int[nnz];
    int* iat_loc = new int[nn + 1];

    // Copy the data from i_idx and j_idx to the CSR arrays
    for (int i = 0; i < nnz; i++) {
        ja_loc[i] = j_idx[i];
    }

    // Compute the CSR arrays from i_idx and ja_loc
    irow2iat(nn, nnz, &i_idx[0], iat_loc);

    H->set_n_term(nnz);
    H->set_ncol(nn);
    H->copy_ja(ja_loc);
    H->copy_iat(iat_loc);
    // H->printComponents();
    delete[] ja_loc;
    delete[] iat_loc;
}

double get_loc(double** coord, double** topol, double *D, double*V, int el, double*** Loc){
    double *t = topol[el];
    double x[3];  //i,j,m
    double y[3];  //i,j,m
    double a[3],b[3],c[3], delta, temp, mod_v;

    for(int i = 0; i<3; i++){
        x[i] = coord[static_cast<int>(t[i]-1)][0];
        y[i] = coord[static_cast<int>(t[i]-1)][1];
    }

    a[0] = x[1]*y[2] - x[2]*y[1];
    a[1] = x[2]*y[0] - x[0]*y[2];
    a[2] = x[0]*y[1] - x[1]*y[0];

    int idx1, idx2;
    for(int i = 0; i<3;i++){
        idx1 = (i+1)%3;
        idx2 = (i+2)%3;
        b[i] = y[idx1] - y[idx2];
        c[i] = x[idx2] - x[idx1];
    }

    delta = (a[0]+a[1]+a[2])/2;
    mod_v = norm(V,2);

    if(mod_v>eps){// prevent zero division
        double temp;
        for(int i = 0; i<3; i++){
            for(int j = 0; j<3; j++){
                temp = (V[0]*b[j] + V[1]*c[j]);
                Loc[0][i][j] = (D[0]*b[i]*b[j]+ D[1]*c[i]*c[j])/(4*delta);
                Loc[1][i][j] = temp/6;
                Loc[2][i][j] = (V[0]*b[i] + V[1]*c[i])*temp/(8*delta*mod_v*max(D,2));
            }
        }
    }else{
        std::cout<<" Norm v zero, just compute stiffness Matrix H \n";
        for(int i = 0; i<3; i++){
            for(int j = 0; j<3; j++){
                Loc[0][i][j] = (D[0]*b[i]*b[j]+ D[1]*c[i]*c[j])/(4*delta);
            }
        }
    }


    return delta;
}

void loc2glob(sparse* A_ptr, double** Loc, double** topol, int e){
    double* coef = A_ptr->get_coef();
    int* ja = A_ptr->get_ja();
    int* iat = A_ptr->get_iat();
    int row, col;
    double* top = topol[e];
    for(int i = 0; i<3; i++){   // choose row
        row = static_cast<int>(top[i]) - 1;
        for(int j = 0; j<3; j++){// chose col
            col = static_cast<int>(top[j]) - 1;
            int k;
            for(k = iat[row]; k<iat[row+1];k++){ //go over column indices in ja
                if(ja[k]==col){break;}
            }
            #pragma omp atomic
            coef[k] += Loc[i][j];
        }
    }
}

sparse* create_FEM_mat(double** coord, double** topol, int ne, int nn, double* D,\
                        double h, double tau, double* V, double* delta){
    sparse* mat_list; 
    mat_list = new sparse[3]; // H,B,S
    double*** Loc = (double***) malloc(3*sizeof(double**));
	// double ***Loc;
	// Loc = new double**[3];
    int buffer_length = static_cast<int>(nn * nn * 0.1);

    int nterm, nr;

    int* ja = (int*) calloc(buffer_length,sizeof(int));
    int* iat = (int*) calloc(nn+1,sizeof(int));

    //allocate space for the local H, B, S
    for(int i = 0; i<3;i++){Loc[i] = mat_allocation(3,3);}

    char fname[] = "fullmat.txt";
    // stiffness_struct(fname, ne,  nn,  topol);
    // createMat(&mat_list[0], fname);
    stiffness_struct_small_loops(&mat_list[0], ne,  nn,  topol);

    mat_list[2] = mat_list[1] = mat_list[0];

    for(int i = 0; i<ne; i++){
        delta[i] = get_loc(coord,topol,D,V,i,Loc);
        for(int k = 0; k<3;k++){
            loc2glob(&mat_list[k],Loc[k],topol,i);
        }
    }
    // printVec(delta,ne);
    mat_list[2].scalarMult(tau*h);

    return mat_list;
}


double force(double x, double y){
    double f = 0;
    // do stuff
    return f;
}


void imposeBC(sparse* H, double** bound, double* q, int nb){
    double R = 1e15;
    int j, bound_idx, start, end;
    int* iat = H->get_iat();
    int* ja = H->get_ja();
    double* coef = H->get_coef();

    for(int i = 0; i<nb; ++i){
        bound_idx = static_cast<int>(bound[i][0])-1;
        start = iat[bound_idx];
        end = iat[bound_idx+1];
        
        for(j = start; j<end; j++){
            if (ja[j] == bound_idx){break;}
        }

        coef[j] = R;
        q[bound_idx] = R*bound[i][1];
    }
}



double* getAssembledSystem(sparse* A){
    // FEM assembly
	char mesh_path[] = "mesh/mesh.dat";
	char node_path[] = "mesh/dirnod.dat";
	char coord_path[] = "mesh/xy.dat";
	
	double D[2] = {1,1};
	double V[2] = {1,3};
	int ne, nn, nb; // number of mesh (800), number of nodes(441), number of ? (80)

	double **coord, **bound, *delta, *delta_node, *F;
	double **topol, *temp, *q;

	sparse* FEM;


	ifstream mesh(mesh_path);
	ifstream nodes(node_path);
	ifstream xy(coord_path);

	//read data
	int not_used;

	xy >> nn >> not_used;
	nodes >> nb >> not_used;
	mesh >> ne >> not_used >> not_used >> not_used;

	bound = mat_allocation(nb,2);
	coord = mat_allocation(nn,2);
	topol = mat_allocation(ne,3);

    

	for (int i = 0; i < nn; ++i){xy >> coord[i][0] >> coord[i][1];}
	for (int i = 0; i < nb; ++i){nodes >> bound[i][0] >> bound[i][1];}
	for (int i = 0; i < ne; ++i){mesh >> topol[i][0] >> topol[i][1] >> topol[i][2]>>not_used;}


	// printMat(coord,nb,2);

	xy.close();
	nodes.close();
	mesh.close();

	double h = sqrt(2)*(coord[0][1]-coord[1][1]);// why sqrt2
	double tau = 0.1; // whats tau?

	delta = (double*)malloc(ne*sizeof(double));

	delta_node = (double*)calloc(nn,sizeof(double));

	F = (double*)malloc(nn*sizeof(double));
	q = (double*)calloc(nn,sizeof(double));



	FEM = create_FEM_mat(coord, topol, ne, nn, D, h, tau, V, delta);

	int idx;
    #pragma omp parallel for schedule(dynamic,nn/1000+1)
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

	*A = FEM[0];
	vector_update(q,F,1,nn);

    mat_free(bound,nb,2);
    mat_free(coord,nn,2);
    mat_free(topol,ne,3);
    free(delta);
    free(delta_node);
    free(F);
    delete[] FEM;


    return q;
}
