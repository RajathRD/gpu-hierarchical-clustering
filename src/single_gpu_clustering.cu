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
#define PRINT_LOG 0
#define PRINT_ANALYSIS 0
/* Define constants */
#define RANGE 100

/*****************************************************************/

// Function declarations
void  seq_clustering(float *, unsigned int, unsigned int, int *, float *);
//void  gpu_clustering(float *, unsigned int, unsigned int);
void calculate_pairwise_dists(float *, int, int, float *);
void find_pairwise_min(float *, int, float *, int *);
void merge_clusters(int *, int, int, int);
float calculate_dist(float *, int, int, int);
void print_float_matrix(float *, int, int);
void print_int_matrix(int *, int, int);
int get_parent(int, int *);
// Kernel functions
//__global__ void calculateMatrix(float * temp_d, float * playground_d, unsigned int N);

// Helper functions
void print_float_matrix(float * a, int n, int m){
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++)
      printf("%f ", a[index(i, j, m)]);
    printf("\n");
  }
}

void print_int_matrix(int * a, int n, int m){
  for (int i=0; i<n; i++){
    for(int j=0; j<m; j++)
      printf("%d ", a[index(i,j,m)]);
    printf("\n");
  }
}

void load_data(float * dataset, int n, int m) {
  srand((unsigned int) 0);
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j < m; j++) {
      // assign numbers between 0 and RANGE
      dataset[index(i, j, m)] = ((float)rand()/(float)(RAND_MAX)) * RANGE - RANGE/2.0;
    } 
  }
  if (PRINT_LOG){
    printf("Dataset:\n");
    print_float_matrix(dataset, n, m);
  }
}


/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  //Define variables
  //unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = atoi(argv[3]); // CPU or GPU
  int n = atoi(argv[1]);
  int m = atoi(argv[2]);

  printf("Hierarchical Clustering:\n");
  printf("Dataset size: %d x %d\n", n, m);
  printf("Device Type: %d\n", type_of_device);
  
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
  dataset = (float *)calloc(n*m, sizeof(float));
  if( !dataset )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", n, m);
   exit(1);
  }
  load_data(dataset, n, m);
  printf("Data loaded!\n");
  
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
  float dendrogram[(n-1)*3];
  int * result;
  result = (int *)calloc(n, sizeof(int));
  if( !type_of_device ) // The CPU sequential version
  {  
    start = clock();
    seq_clustering(dataset, n, m, result, dendrogram);    
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
void  seq_clustering(float * dataset, unsigned int n, unsigned int m, int* result, float * dendrogram)
{
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;

  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  
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

  // O(n*n*m) -> GPU
  start = clock();
  calculate_pairwise_dists(dataset, n, m, dist_matrix);
  end = clock();

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  if (PRINT_ANALYSIS)
    printf("Time taken for distance computation: %lf\n", time_taken);
  
  for (int iteration=0; iteration < n - 1; iteration++) {
    
    float entry[3]; 
    // O(I*n*n) -> GPU
    start = clock();
    find_pairwise_min(dist_matrix, n, entry, result);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    if (PRINT_ANALYSIS)
      printf("Time taken for pairwise min, Iteration %d: %lf\n", iteration, time_taken);
    dendrogram[index(iteration, 0, 3)] = entry[0];
    dendrogram[index(iteration, 1, 3)] = entry[1];
    dendrogram[index(iteration, 2, 3)] = entry[2];
    // O(I*n) -> amortized O(I)
    start = clock();
    merge_clusters(result, (int)entry[0], (int)entry[1], n);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    if (PRINT_ANALYSIS)
      printf("Time taken for merge cluster, Iteration %d: %lf\n", iteration, time_taken);
    if (PRINT_LOG){
      printf("Iteartion #%d\n", iteration);
      printf("Min Indices: %d, %d\n", (int)entry[0], (int)entry[1]);
      print_int_matrix(result, 1, n);
    }
    
  }

  for (int i=0; i<n; i++) result[i] = get_parent(i, result);

  if (PRINT_LOG){
    printf("Cluster IDs:\n");
    print_int_matrix(result, 1, n);
    printf("Dendrogram:\n");
    print_float_matrix(dendrogram, n-1, 3);
  }

  free(dist_matrix);
  //num_bytes = N*N*sizeof(float);
  /* Copy initial array in temp */
  //memcpy((void *)temp, (void *) playground, num_bytes);
  /* Move new values into old values */ 
  //memcpy((void *)playground, (void *) temp, num_bytes);
}

void calculate_pairwise_dists(float * dataset, int n, int m, float * dist_matrix) {
  // O(n)
  // for (int i = 0; i < n*n; i++) dist_matrix[i] = FLT_MAX;
  
  // O(n*n*m)
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // O(m)
      dist_matrix[index(i, j, n)] = calculate_dist(dataset, i, j, m);
    }
  }  
}

// passing vec1_i and vec2_i instead of float * as dist_matrix is 1-D
float calculate_dist(float * dataset, int i, int j, int dim) {
  float dist = 0;
  // O(m)
  for (int mi = 0; mi < dim; mi++){
    dist += (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]) * (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]);
  }
  return dist;
}


int get_parent(int curr_parent, int* parents) {
  if (parents[curr_parent] == curr_parent) return curr_parent;
  parents[curr_parent] = get_parent(parents[curr_parent], parents);
  return parents[curr_parent];
  // return get_parent(parents[curr_parent], parents);
}


void find_pairwise_min(float * dist_matrix, int n, float* entry, int* parents) {
  entry[0] = 0;
  entry[1] = 0;
  entry[2] = FLT_MAX;
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      if (get_parent(i, parents) != get_parent(j, parents)){
      // if (parents[i] != parents[j]) {
        float curr_dist = dist_matrix[index(i, j, n)];
        if (curr_dist < entry[2]) {
          entry[0] = i;
          entry[1] = j;
          entry[2] = curr_dist;
        }
      }
    }
  }
}


void merge_clusters(int * result, int data_point_i, int data_point_j, int dim) {
  if (!(data_point_i >= 0 && data_point_i < dim && data_point_j >= 0 && data_point_j < dim)) {
    printf("merge_clusters out of bounds");
    return;
  } 
  // int cluster_j = result[data_point_j];
  // for(int i=0; i<dim; i++)
  //   if(result[i] == cluster_j)
  //     result[i] = result[data_point_i];
  int parent_i = get_parent(data_point_i, result);
  result[get_parent(data_point_j, result)] = parent_i;
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

