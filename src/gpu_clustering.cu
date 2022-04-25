#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
#include <math.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/* Config params */
#define PRINT_LOG 0
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
__global__ void find_pairwise_min_cuda(float * dist_matrix_d, int n, int * indices);
__global__ void remove_cluster(float * dist_matrix_d, int right_cluster, int n);
__global__ void update_cluster(float * dist_matrix_d, int left_cluster, int right_cluster, int n);

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
  
  if (PRINT_LOG) {
    printf("Dendrogram:\n");
    print_float_matrix(dendrogram, n-1, 3);
  }
  printf("Time taken for GPU %lf\n", time_taken);
  
  free(dataset);

  return 0;
}

/***************** The GPU version *********************/
void gpu_clustering(float * dataset, unsigned int n, unsigned int m, float * dendrogram){
  double time_taken;
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
  int * indices;
  cudaMalloc((void**) &indices, n*n*sizeof(int));
  if (!indices) {
    fprintf(stderr, " Cannot allocate cuda indices %u array\n", n*n);
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
  int block_cnt = (int) ceil((float)n*n / thread_cnt);
  printf("Launching kernel with %d blocks and %d threads\n", block_cnt, thread_cnt);

  // O(1)
  start = clock();
  calculate_pairwise_dists_cuda<<<block_cnt, thread_cnt>>>(dataset_d, dist_matrix_d, n, m);
  cudaDeviceSynchronize();
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

  // O(n)
  for (int iteration=0; iteration < n - 1; iteration++) {
    // printf("\n\n --> iteration = %d\n", iteration);

    // O(log n)
    find_pairwise_min_cuda<<<block_cnt, thread_cnt>>> (dist_matrix_d, n, indices);
    cudaDeviceSynchronize();

    // Move min value index to host memory
    int* min_val_idx;
    if(!(min_val_idx = (int *) malloc(sizeof(int)))) {
      printf("Error allocating min_val_idx\n");
      exit(1);
    }
    cudaMemcpy(min_val_idx, indices, sizeof(int), cudaMemcpyDeviceToHost);

    // Move min value to host memory
    float* min_val;
    if(!(min_val = (float *) malloc(sizeof(float)))) {
      printf("Error allocating min_val\n");
      exit(1);
    }
    cudaMemcpy(min_val, (dist_matrix_d+*min_val_idx), sizeof(float), cudaMemcpyDeviceToHost);

    float min_value = *min_val;
    int i = *min_val_idx/n;
    int j = *min_val_idx%n;

    // Deallocated memories used to move results from device to host
    free(min_val);
    free(min_val_idx);

    // Always i should be smaller than j - cluster with higher index gets merged to the cluster with lower index
    if (i > j) {
      int temp = j;
      j = i;
      i = temp;
    }

    // printf("--> i %d, j %d, min_val %.2f\n", i, j, min_value);

    dendrogram[index(iteration, 0, 3)] = (float) i;
    dendrogram[index(iteration, 1, 3)] = (float) j;
    dendrogram[index(iteration, 2, 3)] = min_value;

    // O(1) - Update left cluster's distance with all others
    update_cluster<<<block_cnt, thread_cnt>>> (dist_matrix_d, i, j, n);
    cudaDeviceSynchronize();
    if (PRINT_LOG) {
      printf("Update left cluster's distance with all others: Dist Matrix:\n");
      cudaMemcpy(dist_matrix, dist_matrix_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
      print_float_matrix(dist_matrix, n, n);
    }

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
  cudaFree(indices);
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
    int i = index / n;
    int j = index % n;
    if (i<n && j < n) {
      if (i == j) dist_matrix[index(i, j, n)] = FLT_MAX;
      else {
        float dist = 0;
        for(int mi=0; mi<m; mi++){
          float x = (dataset[index(i, mi, m)] - dataset[index(j,mi,m)]);
          dist += x * x;
        }
        dist_matrix[index(i, j, n)] = dist;
      }
    }
  }
}

/*
  Finds minimum index of a minimum element in an array
*/
__global__ void find_pairwise_min_cuda(float * dist_matrix_d, int n, int * indices) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int stride = n*n/2;
  while (true) {
    __syncthreads();
    if (index <= stride) {
      int left_idx = index;
      int right_idx = index + stride + 1;

      float left_val = (stride == n*n/2) ? dist_matrix_d[left_idx] : dist_matrix_d[indices[left_idx]];
      // We can be outside of boundary in first iteration, handle it gracefully
      float right_val = FLT_MAX;
      if (right_idx < n*n) {
        right_val = (stride == n*n/2) ? dist_matrix_d[right_idx] : dist_matrix_d[indices[right_idx]];
      }

      // printf("find_pairwise_min_cuda - left_idx %d, indices[left_idx] %d, left_val %.2f and right_idx %d, indices[right_idx] %d, right_val %.2f | index %d, stride %d, n %d\n", 
      // left_idx, indices[left_idx], left_val, right_idx, indices[right_idx], right_val, index, stride, n);

      if (left_val <= right_val) {
        indices[left_idx] = (stride == n*n/2) ? left_idx : indices[left_idx];
      } else {
        indices[left_idx] = (stride == n*n/2) ? right_idx : indices[right_idx];
      }

      // printf("find_pairwise_min_cuda (answer) - left_idx %d, indices[left_idx] %d, dist_matrix_d[indices[left_idx]] %.2f\n", left_idx, indices[left_idx], dist_matrix_d[indices[left_idx]]);
    }

    // Do last check when there are just 2 elements in the array (when stride is 0)
    if (stride == 0) break;
    stride /= 2;
  }
}