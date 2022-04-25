/* 
 *
 * FIXME: Put docs here
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
#define swap(a,b)   {a^=b; b^=a; a^=b;}
/* Config params */
#define PRINT_LOG 0
#define PRINT_DENDRO 1
#define PRINT_ANALYSIS 0

/* Define constants */
#define RANGE 100

/**************************** Definitions *************************************/

// Function declarations
void seq_clustering(float *, unsigned int, unsigned int, float *);
void gpu_clustering(float *, unsigned int, unsigned int, float *);
void calculate_pairwise_dists(float *, int, int, float *);
void find_pairwise_min(float *, int, float *, int *);
void merge_clusters(int *, int, int, int);
float calculate_dist(float *, int, int, int);
void print_float_matrix(float *, int, int);
void print_int_matrix(int *, int, int);
int get_parent(int, int *);

// Kernel functions
__global__ void calculate_pairwise_dists_cuda(float *, float *, unsigned int, unsigned int);
__global__ void remove_cluster(float * dist_matrix_d, int right_cluster, int n);
__global__ void update_cluster(float * dist_matrix_d, int left_cluster, int right_cluster, int n);
__global__ void min_reduction(float *, unsigned int *, unsigned int *, unsigned int);

/*************************** Helper Functions **************************************/
void print_float_matrix(float * a, int n, int m){
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++) {
      float curr = a[index(i, j, m)];
      if ((fabs(FLT_MAX - curr)) < 0.000001) {
        printf("FLT_MAX ");
      } else {
        printf("%.2f ", a[index(i, j, m)]);
      }
    }
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

void print_dendro(float * dendrogram, unsigned int n){
  printf("Dendrogram:\n");
  for(int i=0; i<n-1; i++){
      printf("I: %d -- (%.0f <- %.0f) : %.2f\n", i+1, dendrogram[index(i, 0, 3)], dendrogram[index(i, 1, 3)], dendrogram[index(i, 2, 3)]);
  }
}

// void load_test_data(float * dataset) {
//   float arr[6][2] = {
//     {0.0,0.0},
//     {1.0,1.0},
//     {10.0,10.0},
//     {11.0,11.0},
//     {-100.0,-100.0},
//     {-111.0,111.0}};

//   int n = 6;
//   int m = 2;

//   for (int i = 0; i < n; i ++) {
//     for (int j = 0; j < m; j++) {
//       dataset[index(i, j, m)] = arr[i][j];
//     } 
//   }

//   if (PRINT_LOG){
//     printf("Dataset:\n");
//     print_float_matrix(dataset, n, m);
//   }
// }


/**************************** main() *************************************/
int main(int argc, char * argv[])
{
  //Define variables
  //unsigned int N; /* Dimention of NxN matrix */
  
  int n = atoi(argv[1]);
  int m = atoi(argv[2]);

  printf("Hierarchical Clustering:\n");
  printf("Dataset size: %d x %d\n", n, m);
  
  
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
  float * dataset = (float *)calloc(n*m, sizeof(float));
  if(!dataset) {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", n, m);
   exit(1);
  }

  load_data(dataset, n, m);
  //load_test_data(dataset);
  printf("Data loaded!\n");
  


  float dendrogram[(n-1)*3];

// The GPU version
  start = clock();
  gpu_clustering(dataset, n, m, dendrogram);
  end = clock();    
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", "GPU", time_taken);

  if (PRINT_DENDRO){
    print_dendro(dendrogram, n);
  }
  
  free(dataset);

  return 0;
}

/***************** The GPU version *********************/
void gpu_clustering(float * dataset, unsigned int n, unsigned int m, float * dendrogram){
  double time_taken;
  unsigned int size;
  clock_t start, end;

  // FIXME: Remove this in final cleanup, here only for testing
  float* dist_matrix = (float *)calloc(n*n, sizeof(float));
  if( !dist_matrix ) {
   fprintf(stderr, " Cannot allocate dist_matrix %u array\n", n*n);
   exit(1);
  }

  float * dist_matrix_d;
  cudaMalloc((void**) &dist_matrix_d, n*n*sizeof(float));
  if (!dist_matrix_d) {
    fprintf(stderr, " Cannot allocate cuda dist_matrix_d %u array\n", n*n);
    exit(1);
  }

  // Temp array used by kernel function to find element in a distance matrix
  unsigned int* indices = (unsigned int *)calloc(n*n, sizeof(unsigned int));
  unsigned int* indices_ptr = &indices[0];
  // TODO: check for failure of alloc
  for (int i=0; i<n*n; i++) indices[i]=i;

  unsigned int * indices_d;
  cudaMalloc((void**) &indices_d, n*n*sizeof(unsigned int));
  if (!indices_d) {
    fprintf(stderr, " Cannot allocate cuda indices_d %u array\n", n*n);
    exit(1);
  }

  float * dataset_d;
  cudaMalloc((void**) &dataset_d, n*m*sizeof(float));
  if (!dataset_d) {
    fprintf(stderr, " Cannot allocate cuda dataset %u array\n", n*n);
    exit(1);
  }
  cudaMemcpy(dataset_d, dataset, n*m*sizeof(float), cudaMemcpyHostToDevice);

  // Maximum number of threads per block in cuda1.cims.nyu.edu 
  int thread_cnt = 1024;
  int block_cnt = (int) ceil(n*n / (double)thread_cnt);
  printf("Launching kernel with %d blocks and %d threads\n", block_cnt, thread_cnt);

  // O(1)
  start = clock();
  calculate_pairwise_dists_cuda<<<block_cnt, thread_cnt>>>(dataset_d, dist_matrix_d, n, m);
  // cudaDeviceSynchronize();
  if (PRINT_LOG) {
    printf("Dist Matrix:\n");
    cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    print_float_matrix(dist_matrix, n, n);
  }
  end = clock();

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  if (PRINT_ANALYSIS)
    printf("Time taken for distance computation: %lf\n", time_taken);
  
  start = clock();

  // TODO: check for failure
  thread_cnt = 1024;
  block_cnt = (int) ceil(n*n / (double)thread_cnt);
  unsigned int * block_mins = (unsigned int *) calloc (block_cnt, sizeof(unsigned int));
  unsigned int * block_mins_d;
  cudaMalloc((void**)&block_mins_d, block_cnt*sizeof(unsigned int));
  // O(n)
  for (int iteration=0; iteration < n - 1; iteration++) {
    // printf("\n\n --> iteration = %d\n", iteration);
    size = n*n;
    block_cnt = (int) ceil( n*n / (double)thread_cnt);
    indices_ptr = indices;
    
    while (size > 1){
      block_cnt = (int)ceil((float)size/thread_cnt);
      cudaMemcpy(indices_d, indices_ptr, size*sizeof(unsigned int), cudaMemcpyHostToDevice);
      min_reduction<<<block_cnt, thread_cnt>>>(dist_matrix_d, indices_d, block_mins_d, size);
      cudaMemcpy(block_mins, block_mins_d, block_cnt*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      indices_ptr = block_mins;
      size = block_cnt;
    }
    
    unsigned int min_val_idx = block_mins[0];
    float min_value = dist_matrix[min_val_idx];
    int i = min_val_idx/n;
    int j = min_val_idx%n;

    // Always i should be smaller than j - cluster with higher index gets merged to the cluster with lower index
    if (i > j) swap(i,j)

    // printf("--> i %d, j %d, min_val %.2f\n", i, j, min_value);

    dendrogram[index(iteration, 0, 3)] = (float) i;
    dendrogram[index(iteration, 1, 3)] = (float) j;
    dendrogram[index(iteration, 2, 3)] = min_value;
    
    thread_cnt = 1024;
    block_cnt = (int) ceil(n*n / (double)thread_cnt);
    // O(1) - Update left cluster's distance with all others
    update_cluster<<<block_cnt, thread_cnt>>> (dist_matrix_d, i, j, n);
    cudaDeviceSynchronize();
    if (PRINT_LOG) {
      printf("Update left cluster's distance with all others: Dist Matrix:\n");
      cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
      print_float_matrix(dist_matrix, n, n);
    }

    thread_cnt = 1024;
    block_cnt = (int) ceil(n*n / (double)thread_cnt);    
    
    // O(1) - Remove right clusters from further consideration
    remove_cluster<<<block_cnt, thread_cnt>>>(dist_matrix_d, j, n);
    cudaDeviceSynchronize();
    if (PRINT_LOG) {
      printf("Remove right clusters from further consideration: Dist Matrix:\n");
      cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
      print_float_matrix(dist_matrix, n, n);
    }
    
  }
  cudaFree(dataset_d);
  cudaFree(indices_d);
  cudaFree(dist_matrix_d);
}

/*
  Right is being merged to left
  So remove all distance entries for right with any other cluster 
*/ 
__global__ void update_cluster(float * dist_matrix_d, int left_cluster, int right_cluster, int n) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= n*n) return;

  int i = index/n;
  int j = index%n;

  if (i == j) return;

  if (i == left_cluster) {
    float new_min = min(dist_matrix_d[index(i, j, n)], dist_matrix_d[index(right_cluster, j, n)]);
    dist_matrix_d[index(i, j, n)] = new_min;
    // printf("update_cluster - i == left_cluster | i: %d, j: %d, n: %d, new_min: %.2f\n", i, j, n, new_min);
  } else if (j == left_cluster) {
    float new_min = min(dist_matrix_d[index(i, j, n)], dist_matrix_d[index(i, right_cluster, n)]);
    dist_matrix_d[index(i, j, n)] = new_min;
    // printf("update_cluster - j == left_cluster | i: %d, j: %d, n: %d, new_min: %.2f\n", i, j, n, new_min);
  }
}

/*
  Right is being merged to left
  So remove all distance entries for right with any other cluster 
*/ 
__global__ void remove_cluster(float * dist_matrix_d, int right_cluster, int n) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= n*n) return;

  int i = index/n;
  int j = index%n;
    // printf("remove_cluster | i: %d, j: %d, index: %d, dist_matrix_d[index]: %.2f, right_cluster: %d\n", i, j, index, dist_matrix_d[index], right_cluster);
  if (i == right_cluster || j == right_cluster) {
    dist_matrix_d[index] = FLT_MAX;
    // printf("remove_cluster - i == right_cluster || j == right_cluster | i: %d, j: %d, index: %d, dist_matrix_d[index]: %.2f, right_cluster: %d\n", i, j, index, dist_matrix_d[index], right_cluster);
  }
}

__global__ void calculate_pairwise_dists_cuda(float * dataset, float * dist_matrix, unsigned int n, unsigned int m)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  // Dont update if thread id is outside of the box
  if (index < n*n){
    int r = index / n;
    int c = index % n;
    if (r<n && c < n) {
      if (r >= c) dist_matrix[index(r, c, n)] = FLT_MAX;
      else {
        float dist = 0;
        for(int mi=0; mi<m; mi++){
          float x = (dataset[index(r, mi, m)] - dataset[index(c,mi,m)]);
          dist += x * x;
        }
        dist_matrix[index(r, c, n)] = dist;
      }
    }
  }
}

__global__ void min_reduction(float *arr, unsigned int * indices, unsigned int * block_mins, unsigned int n){
  unsigned int offset = blockDim.x * blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x / 2;
  int left = index, right = left + stride;

  if (index < n){
    while (stride > 0){
      left = offset + tid;
      right = left + stride;
      
      if (tid < stride && right < n){
          // printf("stride: %d, left: %d, right: %d\n", stride, left, right);
          // printf("a[%d]=%.2f, a[%d]=%.2f\n", indices[left], arr[indices[left]], indices[right], arr[indices[right]]);
          if(arr[indices[left]] > arr[indices[right]]){
              indices[left] = indices[right];
          }
      }
      stride /= 2;
      __syncthreads();
    }
  }

  if (tid == 0){
    block_mins[blockIdx.x] = indices[offset];
  }
}

// shared_memory
// __global__ void min_reduction(float *arr, unsigned int * indices, unsigned int * block_mins, unsigned int n){
//   extern __shared__ unsigned int sindices[];
//   unsigned int offset = blockDim.x * blockIdx.x;
//   unsigned int tid = threadIdx.x;
//   unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x / 2;
//   int left = index, right = left + stride;
  
//   sindices[tid] = indices[index];
//   __syncthreads();

//   if (index < n){
//     while (stride > 0){
//         left = tid;
//         right = left + stride;
//         if (tid < stride && right < n){
//             // printf("stride: %d, left: %d, right: %d\n", stride, left, right);
//             // printf("a[%d]=%.2f, a[%d]=%.2f\n", sindices[left], arr[sindices[left]], sindices[right], arr[sindices[right]]);

//             if(arr[sindices[left]] > arr[sindices[right]]){
//                 sindices[left] = sindices[right];
//             }
//         }
//       stride /= 2;
//       __syncthreads();
//     }
//   }
//   if (tid == 0){
//     block_mins[blockIdx.x] = sindices[0];
//   }
// }

// __global__ void find_pairwise_min_cuda(float * dist_matrix_d, int n, int * indices) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;

//   int stride = blockDim.x/2;
//   while (true) {
//     __syncthreads();
//     if (index <= stride) {
//       int left_idx = index;
//       int right_idx = index + stride + 1;

//       float left_val = (stride == n*n/2) ? dist_matrix_d[left_idx] : dist_matrix_d[indices[left_idx]];
//       // We can be outside of boundary in first iteration, handle it gracefully
//       float right_val = FLT_MAX;
//       if (right_idx < n*n) {
//         right_val = (stride == n*n/2) ? dist_matrix_d[right_idx] : dist_matrix_d[indices[right_idx]];
//       }

//       // printf("find_pairwise_min_cuda - left_idx %d, indices[left_idx] %d, left_val %.2f and right_idx %d, indices[right_idx] %d, right_val %.2f | index %d, stride %d, n %d\n", 
//       // left_idx, indices[left_idx], left_val, right_idx, indices[right_idx], right_val, index, stride, n);

//       if (left_val <= right_val) {
//         indices[left_idx] = (stride == n*n/2) ? left_idx : indices[left_idx];
//       } else {
//         indices[left_idx] = (stride == n*n/2) ? right_idx : indices[right_idx];
//       }

//       // printf("find_pairwise_min_cuda (answer) - left_idx %d, indices[left_idx] %d, dist_matrix_d[indices[left_idx]] %.2f\n", left_idx, indices[left_idx], dist_matrix_d[indices[left_idx]]);
//     }

//     // Do last check when there are just 2 elements in the array (when stride is 0)
//     if (stride == 0) break;
//     stride /= 2;
//   }
// }


/*
 1. Improve CPU version to n^2 long n or n^2 
 2. GPU get pairwise min parallel reduction
 3. Merging/updating matrices

 4. Testing/Validation - till GPU runs out of memory
 5. Improvements

  Tasks:
    - DONE: Fix issues for these:
      ./single_gpu_clustering 7 1 1
      ./single_gpu_clustering 10 1 1
      ./single_gpu_clustering 7 2 1
  
    - TODO: Update arg checks for 4 inputs as well as for tests
    - TODO: Add tester in a separate file with sample tests for GPU version

  Notes:
    - There are 3 cuda memory (cudaMalloc with int and float types) allocations so, total memory needed is 4*(2*n*n + n*m). 
      Sine single device memory is 60GB, n should be tested as much as 120000 before hitting memory limits. 

*/
