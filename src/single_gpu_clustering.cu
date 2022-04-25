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
#define PRINT_LOG 1
#define PRINT_ANALYSIS 0

/* Define constants */
#define RANGE 100

/**************************** Definitions *************************************/

// Function declarations
void gpu_clustering(float *, unsigned int, unsigned int, float *);
void print_float_matrix(float *, int, int);
void print_int_matrix(int *, int, int);

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
  if (PRINT_ANALYSIS){
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


/**************************** main() *************************************/
int main(int argc, char * argv[])
{
  //Define variables  
  int n = atoi(argv[1]);
  int m = atoi(argv[2]);

  if (PRINT_ANALYSIS) {
    printf("Hierarchical Clustering:\n");
    printf("Dataset size: %d x %d\n", n, m);
  }
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  // Validate
  if(argc != 3)
  {
    fprintf(stderr, "usage: program n m\n");
    fprintf(stderr, "n = dimension n\n");
    fprintf(stderr, "m = dimension m\n");
    exit(1);
  }

  //Load data
  float * dataset = (float *)calloc(n*m, sizeof(float));
  if(!dataset) {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", n, m);
   exit(1);
  }
  load_data(dataset, n, m);

  float dendrogram[(n-1)*3];

  // The GPU version
  start = clock();
  gpu_clustering(dataset, n, m, dendrogram);
  end = clock();    
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", "GPU", time_taken);

  if (PRINT_LOG){
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

  if (PRINT_ANALYSIS) {
    printf("Dist Matrix:\n");
    cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    print_float_matrix(dist_matrix, n, n);
  }
  end = clock();

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  if (PRINT_ANALYSIS)
    printf("Time taken for distance computation: %lf\n", time_taken);
  
  start = clock();

  thread_cnt = 128;
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
      min_reduction<<<block_cnt, thread_cnt, thread_cnt*sizeof(unsigned int)>>>(dist_matrix_d, indices_d, block_mins_d, size);
      cudaMemcpy(block_mins, block_mins_d, block_cnt*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      indices_ptr = block_mins;
      size = block_cnt;
      // printf("Block Mins: ");
      // for(int i=0; i<block_cnt; i++) printf("%d ", block_mins[i]);
      // printf("\n");
    }
    
    unsigned int min_val_idx = block_mins[0];
    float min_value = dist_matrix[min_val_idx];
    int i = min_val_idx/n;
    int j = min_val_idx%n;
    
    // Always i should be smaller than j - cluster with higher index gets merged to the cluster with lower index
    if (i > j) swap(i,j)

    // printf("--> i %d, j %d, min_val %.2f\n", i, j, min_value);
    // break;
    dendrogram[index(iteration, 0, 3)] = (float) i;
    dendrogram[index(iteration, 1, 3)] = (float) j;
    dendrogram[index(iteration, 2, 3)] = min_value;
    
    thread_cnt = 1024;
    block_cnt = (int) ceil(n*n / (double)thread_cnt);
    // O(1) - Update left cluster's distance with all others
    update_cluster<<<block_cnt, thread_cnt>>> (dist_matrix_d, i, j, n);
    cudaDeviceSynchronize();
    if (PRINT_ANALYSIS) {
      printf("Update left cluster's distance with all others: Dist Matrix:\n");
      cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
      print_float_matrix(dist_matrix, n, n);
    }

    thread_cnt = 1024;
    block_cnt = (int) ceil(n*n / (double)thread_cnt);    
    
    // O(1) - Remove right clusters from further consideration
    remove_cluster<<<block_cnt, thread_cnt>>>(dist_matrix_d, j, n);
    cudaDeviceSynchronize();
    if (PRINT_ANALYSIS) {
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

// shared_memory
__device__ void unroll_last_reduce(float * arr, volatile unsigned int* sindices, int tid) {
  if (arr[sindices[tid]] > arr[sindices[tid+32]]) sindices[tid] = sindices[tid+32];
  if (arr[sindices[tid]] > arr[sindices[tid+16]]) sindices[tid] = sindices[tid+16];
  if (arr[sindices[tid]] > arr[sindices[tid+8]]) sindices[tid] = sindices[tid+8];
  if (arr[sindices[tid]] > arr[sindices[tid+4]]) sindices[tid] = sindices[tid+4];
  if (arr[sindices[tid]] > arr[sindices[tid+2]]) sindices[tid] = sindices[tid+2];
  if (arr[sindices[tid]] > arr[sindices[tid+1]]) sindices[tid] = sindices[tid+1];
}

__global__ void min_reduction(float *arr, unsigned int * indices, unsigned int * block_mins, unsigned int n){
  extern __shared__ unsigned int sindices[];
  unsigned int tid = threadIdx.x;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x / 2;
  int left = index, right = left + stride;
  
  sindices[tid] = indices[index];
  __syncthreads();

  if (index < n){
    while (stride > 32){
        left = tid;
        right = left + stride;
        if (tid < stride && right < n){
            // printf("stride: %d, left: %d, right: %d\n", stride, left, right);
            // printf("a[%d]=%.2f, a[%d]=%.2f\n", sindices[left], arr[sindices[left]], sindices[right], arr[sindices[right]]);

            if(arr[sindices[left]] > arr[sindices[right]]){
                sindices[left] = sindices[right];
            }
        }
      stride /= 2;
      __syncthreads();
    }
  }

  if (tid < 32) unroll_last_reduce(arr, sindices, tid);

  if (tid == 0){
    block_mins[blockIdx.x] = sindices[0];
  }
}

