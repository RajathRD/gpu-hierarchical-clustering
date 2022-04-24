#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <limits.h>
#include <vector>
#include <sstream>
using namespace std;

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)
#define PRINT_LOG 1
/* Define constants */
#define RANGE 100

/*****************************************************************/

// Function declarations
void seq_clustering(float *, unsigned int, unsigned int, int *);
void calculate_pairwise_dists(float *, int, int, float *);
float calculate_dist(float *, int, int, int);
void print_matrix(float *, int, int);
void gen_data(float *, int, int);

// Helper functions
void print_matrix(float * a, int n, int m){
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++)
      printf("%f ", a[index(i, j, m)]);
    printf("\n");
  }
}

void gen_data(float * dataset, int n, int m) {
  srand((unsigned int) 0);
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j < m; j++) {
      // assign numbers between 0 and RANGE
      dataset[index(i, j, m)] = ((float)rand()/(float)(RAND_MAX)) * RANGE - RANGE/2.0;
    } 
  }
}

int main(int argc, char * argv[])
{
  //Define variables
  int N;
  int M;
  float * dataset;
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;

  int * result;

  if(argc == 1)
  {
    printf("Hierarchical Clustering with Unit Tests:\n");
    // We are running unit tests
    vector<string> tests;
    // Read in the names of the test files
    string line;
    ifstream myfile ("unittest/tests.txt");
    if (myfile.is_open()) {
        while ( getline (myfile,line) ) {
            tests.push_back(line);
        }
        myfile.close();
    }
    else {
        cout << "Unable to open file";
    }

    for(int i = 0; i< tests.size(); i++) {
        string test_path = "unittest/tests/" + tests[i];
        // For each test case
        // Read in the data
        int file_count = 0;
        ifstream myfile (test_path.c_str());
        if (myfile.is_open()) {
            while ( getline (myfile, line) ) {
                istringstream iss (line);
                string s;
                if(file_count == 0) {
                    int str_count = 0;
                    while( getline (iss, s, ' ') ) {
                        if(str_count == 0) {
                            N = atoi(s.c_str());
                        }
                        else {
                            M = atoi(s.c_str());
                        }
                        str_count++;
                    }

                    //Malloc dataset
                    dataset = (float *)calloc(N*M, sizeof(float));
                    if( !dataset )
                    {
                        fprintf(stderr, " Cannot allocate the %u x %u array\n", N, M);
                        exit(1);
                    }

                }
                else {
                    int str_count = 0;
                    while( getline (iss, s, ' ') ) {
                        dataset[index(file_count-1, str_count, M)] = atoi(s.c_str());
                        str_count++;
                    }
                }
                file_count++;
            }
            myfile.close();
        }
        else {
            cout << "Unable to open file";
        }

        printf("Test case %d\n", i);
        printf("Dataset size: %d x %d\n", N, M);
        if (PRINT_LOG){
          printf("Dataset:\n");
          print_matrix(dataset, N, M);
        }

        // Allocate result array
        result = (int *)calloc(2*(N-1), sizeof(int));
        if( !result )
        {
            fprintf(stderr, " Cannot allocate result array of size %u\n", 2*(N-1));
            exit(1);
        }

        // Perform clustering
        start = clock();
        seq_clustering(dataset, N, M, result);    
        end = clock();
        
        time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

        //Print Result
        if(PRINT_LOG) {
            printf("Merges made, in order:\n");
            for (int i=0; i<N-1; i++){
                printf("(%d <- %d)\n", result[2*i], result[(2*i)+1]);
            }
        }
        
        printf("Time taken for CPU is %lf\n", time_taken);
        free(result);
    }
  }
  else if(argc == 3) {
    // We are running with simulated data
    N = atoi(argv[1]);
    M = atoi(argv[2]);

    printf("Hierarchical Clustering:\n");
    printf("Dataset size: %d x %d\n", N, M);

    // Generate simulated data
    dataset = (float *)calloc(N*M, sizeof(float));
    if( !dataset )
    {
    fprintf(stderr, " Cannot allocate the %u x %u array\n", N, M);
    exit(1);
    }
    gen_data(dataset, N, M);
    printf("Data loaded!\n");
    if (PRINT_LOG){
      printf("Dataset:\n");
      print_matrix(dataset, N, M);
    }

    result = (int *)calloc(2*(N-1), sizeof(int));
    if( !result )
    {
      fprintf(stderr, " Cannot allocate result array of size %u\n", 2*(N-1));
      exit(1);
    }

    // Perform clustering
    start = clock();
    seq_clustering(dataset, N, M, result);    
    end = clock();
    
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    //Print Result
    if(PRINT_LOG) {
        printf("Merges made, in order:\n");
        for (int i=0; i<N-1; i++){
            printf("(%d <- %d)\n", result[2*i], result[(2*i)+1]);
        }
    }
    
    printf("Time taken for CPU is %lf\n", time_taken);
    free(result);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "To run unit tests: cpu_clustering\n");
    fprintf(stderr, "To run with simulated data: cpu_clustering N M\n");    
    fprintf(stderr, "N = number of data points\n");
    fprintf(stderr, "M = dimension of data points\n");
    exit(1);
  }

  free(dataset);

  return 0;

}

void seq_clustering(float * dataset, unsigned int N, unsigned int M, int* A)
{
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;

  // Initialize A array
  for (int n = 0; n < 2*(N-1); n++) A[n] = 0;

  // Allocate C matrix
  float* C = (float *)calloc(N*N, sizeof(float));
  if( !C )
  {
   fprintf(stderr, " Cannot allocate C %u array\n", N*N);
   exit(1);
  }

  // Initialize C matrix
  for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
          C[index(i, j, N)] = 0;
      }
  }

  // Populate C matrix
  calculate_pairwise_dists(dataset, N, M, C);

  // Allocate I array
  int * I = (int *)calloc(N, sizeof(int));
  if( !I )
  {
   fprintf(stderr, " Cannot allocate I array of length %u\n", N);
   exit(1);
  }

  // Initialize I array
  for(int n = 0; n < N; n++) I[n] = n;

  // Allocate NBM array
  float * NBM = (float *)calloc(2*N, sizeof(float));
  if( !NBM )
  {
   fprintf(stderr, " Cannot allocate NBM array of length %u\n", 2*N);
   exit(1);
  }

  // Initialize NBM array
  for(int n = 0; n < N; n++) {
    int min_dist = INT_MAX;
    int min_idx = 0;
    for(int i = 0; i < N; i++) {
        if(n != i && C[index(n, i, N)] < min_dist) {
            min_dist = C[index(n, i, N)];
            min_idx = i;
        }
    }
    NBM[2*n] = min_dist;
    NBM[(2*n)+1] = min_idx;
  }

  // Clustering
  for(int n = 0; n < N-1; n++) {
    int min_dist = INT_MAX;
    int min_idx = 0;
    for(int i = 0; i < N; i++) {
        if(I[i] == i && NBM[2*i] < min_dist) {
            min_dist = NBM[2*i];
            min_idx = i;
        }
    }
    int i1 = min_idx;
    int i2 = I[(int)NBM[(2*i1)+1]];
    A[2*n] = i1;
    A[(2*n)+1] = i2;
    for(int i = 0; i < N; i++) {
        if(I[i] == i && i != i1 && i != i2) {
            C[index(i, i1, N)] = max(C[index(i1, i, N)], C[index(i2, i, N)]);
            C[index(i1, i, N)] = C[index(i, i1, N)];
        }
        if(I[i] == i2) {
            I[i] = i1;
        }
    }

    min_dist = INT_MAX;
    min_idx = 0;
    for(int i = 0; i < N; i++) {
        if(I[i] == i && i != i1) {
            min_dist = C[index(i1, i, N)];
            min_idx = i;
        }
    }
    NBM[2*i1] = min_dist;
    NBM[(2*i1)+1] = min_idx;
  }
  
  free(C);
  free(I);
  free(NBM);
}

void calculate_pairwise_dists(float * dataset, int N, int M, float * dist_matrix) {
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      dist_matrix[index(i, j, N)] = calculate_dist(dataset, i, j, M);
      dist_matrix[index(j, i, N)] = dist_matrix[index(i, j, N)];
    }
  }  
}

// passing vec1_i and vec2_i instead of float * as dist_matrix is 1-D
float calculate_dist(float * dataset, int i, int j, int dim) {
  float dist = 0;
  for (int mi = 0; mi < dim; mi++){
    dist += (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]) * (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]);
  }
  return dist;
}