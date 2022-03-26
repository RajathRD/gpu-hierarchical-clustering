/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
#include <math.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/* Define constants */
#define RANGE 100

/*****************************************************************/

// Function declarations
void  seq_clustering(float *, unsigned int, unsigned int);
//void  gpu_clustering(float *, unsigned int, unsigned int);
void calculate_pairwise_dists(float * dataset, int n, int m, float * dist_matrix);

// Kernel functions
//__global__ void calculateMatrix(float * temp_d, float * playground_d, unsigned int N);

// Helper functions
void load_data(float * dataset, int n, int m) {
  dataset = (float *)calloc(n*m, sizeof(float));
  if( !dataset )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", n, m);
   exit(1);
  }

  srand((unsigned int) 0);
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j < m; j++) {
      // assign numbers between 0 and RANGE
      dataset[index(i, j, m)] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
    } 
  }
}

void print_dendogram_tree(int * tree, int n) {
}

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  //Define variables
  //unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int n = 10;
  int m = 2;
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  // Validate
  /*if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }*/

  //Load data
  float * dataset;
  load_data(dataset, n, m);

  type_of_device = atoi(argv[3]);
  //N = (unsigned int) atoi(argv[1]);
  //iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  /*playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }*/
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  /*for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  // FIXME: Why N-1? Shouldnt it be N? There is a post about it in Brightspace which has not been answered yet.
  // Will leave it as it is
  for(i = 0; i < N-1; i++)
    playground[index(N-1,i,N)] = 150;
  */

  int * result;
  if( !type_of_device ) // The CPU sequential version
  {  
    start = clock();
    result = seq_clustering(dataset, n, m, result);    
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     end = clock();    
  }
  
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);

  free(dataset);
  free(result);

  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_clustering(float * dataset, int n, int m, int* result)
{
  
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  result = (int *)calloc(n, sizeof(int));
  if( !result )
  {
   fprintf(stderr, " Cannot allocate result %u array\n", n);
   exit(1);
  }
  for (int i = 0; i < n; i++) result[i] = i;

  float* dist_matrix = (float *)calloc(n*n, sizeof(float));
  if( !dist_matrix )
  {
   fprintf(stderr, " Cannot allocate dist_matrix %u array\n", n*n);
   exit(1);
  }

  calculate_pairwise_dists(dataset, n, m, dist_matrix);
  int cluster_size = n;
  while (cluster_size > 0) {
    int indices[2]; 
    find_pairwise_min(dist_matrix, n, indices, result);
    merge_clusters(result, indices[0], indices[1]);
    cluster_size -= 1;
  }
  
  free(dist_matrix);
  //num_bytes = N*N*sizeof(float);
  /* Copy initial array in temp */
  //memcpy((void *)temp, (void *) playground, num_bytes);
  /* Move new values into old values */ 
  //memcpy((void *)playground, (void *) temp, num_bytes);
}

void calculate_pairwise_dists(float * dataset, int n, int m, float * dist_matrix) {
  memset(dist_matrix, FLT_MAX, (size_t) n*n*sizeof(float))
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      dist_matrix[index(i, j, n)] = calculate_dist(dataset[i], dataset[j]);
    }
  }  
}


float calculate_dist(float* vec1, float* vec2, int dim) {
  float dist = 0;
  for (int i = 0; i < dim; i++){
    dist += pow(vec1[i] - vec2[i], 2);
  }
  return sqrt(dist);
}


int get_parent(int i, int* parents) {
  if (parents[i] == i) return i;
  parents[i] = get_parent(parents[i]);
  return parents[i];
}


void find_pairwise_min(float * dist_matrix, int n, int* indices, int* parents) {
  indices[0] = 0;
  indices[1] = 0;
  float min_dist = FLT_MAX;
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      if (get_parent(i, parents) != get_parent(j, parents)) {
        float curr_dist = dist_matrix[index(i, j, n)];
        if (curr_dist < min_dist) {
          min_dist = curr_dist;
          indices[0] = i;
          indices[1] = j;
        }
      }
    }
  }
}


void merge_clusters(int * result, int i, int j, int dim) {
  if (!(i >= 0 && i < dim && j >= 0 && j < dim)) {
    printf("merge_clusters out of bounds");
    return;
  } 

  result[j] = result[i];
} 

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
/*void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{

  size_t num_bytes = N*N*sizeof(float);
  float * playground_d;

  // Move data to device memory
  cudaMalloc((void**) &playground_d, num_bytes);
  cudaMemcpy(playground_d, playground, num_bytes, cudaMemcpyHostToDevice);

  // Maximum number of threads per block in cuda1.cims.nyu.edu 
  int thread_cnt = 1024;
  int block_cnt = (int) ceil((double) N*N / thread_cnt);

  float * temp_d;
  cudaMalloc((void**) &temp_d, num_bytes);
  cudaMemcpy(temp_d, playground_d, num_bytes, cudaMemcpyDeviceToDevice);
  for (int k = 0; k < iterations; k++) 
  {           
    calculateMatrix<<<block_cnt, thread_cnt>>>(temp_d, playground_d, N);
    cudaMemcpy(playground_d, temp_d, num_bytes, cudaMemcpyDeviceToDevice); 
  }

  // Move new values into old values
  cudaMemcpy(playground, temp_d, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(playground_d);
  cudaFree(temp_d);
}

__global__ void calculateMatrix(float * temp_d, float * playground_d, unsigned int N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // Dont update if thread id is outside of the box
  if (idx >= N * N) return;
  int i = idx / N;
  int j = idx % N;
  
  // Dont update the edges
  if (i-1 >= 0 && i+1 < N && j-1 >= 0 && j+1 < N)
  {
    temp_d[index(i,j,N)] = (playground_d[index(i-1,j,N)] +
                            playground_d[index(i+1,j,N)] +
                            playground_d[index(i,j-1,N)] +
                            playground_d[index(i,j+1,N)])/4.0;
  }
} 
*/

/* Helper Functions */
/*void print_matrix(float * matrix, unsigned int N)
{
  for (int i = 0; i < N; i++) 
  {
    for (int j = 0; j < N; j++) 
    {
      float curr = matrix[index(i,j,N)];
      printf("%.2f\t", curr);
    }
    printf("\n");
  }
}
*/

