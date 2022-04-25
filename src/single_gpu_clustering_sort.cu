#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/sort.h>
#include<cuda.h>
#include<time.h> 
#include<math.h>
#include<iostream>

#define index(i, j, N)  ((i)*(N)) + (j)
/* Define constants */
#define PRINT_DENDRO 0
#define RANGE 100

// Host Functions
void  clustering(float *, unsigned int, unsigned int, float *);
void calculate_pairwise_dists(float *, int, int, float *);
float calculate_dist(float *, int, int, int);
// void update_cluster_cpu(unsigned int *, unsigned int, unsigned int, unsigned int);

// Device Functions
// __global__ void calculate_pairwise_dists_cuda(float *, float *, unsigned int, unsigned int);
__global__ void calculate_pairwise_dists_cuda(float *, float *, unsigned int, unsigned int);
__global__ void update_cluster_cuda(unsigned int *, unsigned int, unsigned int, unsigned int);

// Helper functions
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

// Helper functions
void print_thrustfloat_matrix(thrust::host_vector<float> &a, int n, int m){
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

void print_int_matrix(unsigned int * a, unsigned int n, unsigned int m){
  for (int i=0; i<n; i++){
    for(int j=0; j<m; j++)
      printf("%d ", a[index(i,j,m)]);
    printf("\n");
  }
}

void print_thrustint_matrix(thrust::host_vector<unsigned int> &a, unsigned int n, unsigned int m){
  for (int i=0; i<n; i++){
    for(int j=0; j<m; j++)
      printf("%d ", a[index(i,j,m)]);
    printf("\n");
  }
}

void print_dendro(float * dendrogram, int iteration){
  printf("Dendrogram:\n");
  for(int i=0; i<iteration; i++){
      printf("I: %d -- (%.0f <- %.0f) : %.2f\n", i+1, dendrogram[index(i, 0, 3)], dendrogram[index(i, 1, 3)], dendrogram[index(i, 2, 3)]);
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
}

int main(int argc, char * argv[])
{
    double time_taken;
    clock_t start, end;
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    printf("Hierarchical Clustering:\n");
    printf("Dataset size: %d x %d\n", n, m);

    // to measure time taken by a specific part of the code 
    // double time_taken;


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

    float dendrogram[(n-1)*3];
    start = clock();
    clustering(dataset, n, m, dendrogram);   
    end = clock();    
    
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    
    printf("Time taken for %s is %lf\n", "GPU", time_taken);
    return 0;
}

void update_cluster_cpu(unsigned int * cluster, unsigned int cluster_1, unsigned int cluster_2, unsigned int n){
    for (int i=0; i<n; i++){
        if (cluster[i] == cluster_2) cluster[i] = cluster_1;
    }
}

int get_parent(int curr_parent, int* parents) {
  if (parents[curr_parent] == curr_parent) return curr_parent;
  parents[curr_parent] = get_parent(parents[curr_parent], parents);
  return parents[curr_parent];
  // return get_parent(parents[curr_parent], parents);
}

void  clustering(float * dataset, unsigned int n, unsigned int m, float * dendrogram)
{
  // // ith data point belong to cluster[i]
  // unsigned int * cluster = (unsigned int *) calloc (n, sizeof(unsigned int));
  // if( !cluster )
  // {
  //  fprintf(stderr, " Cannot allocate result %u array\n", n);
  //  exit(1);
  // }

  // ith data point belong to cluster[i]
  thrust::host_vector<unsigned int> cluster(n);
  for (int i = 0; i < n; i++) cluster[i] = i;

//   float* dist_matrix = (float *)calloc(n*n, sizeof(float));
//   if( !dist_matrix )
//   {
//    fprintf(stderr, " Cannot allocate dist_matrix %u array\n", n*n);
//    exit(1);
//   }
  
//   std::cout << "CPU Pairwise Dists" << std::endl;
//   calculate_pairwise_dists(dataset, n, m, dist_matrix);
//   print_float_matrix(dist_matrix, n, n);
  
//   std::cout << "Cleared" << std::endl;
//   std::fill(dist_matrix, dist_matrix+n*n, 0);

  // calcualte pairwise distances on GPU
  // std::cout << "GPU Pairwise Dists" << std::endl;
  thrust::host_vector<float> dist_matrix(n*n);
  thrust::device_vector<float> dist_matrix_d(n*n);
  float * dist_matrix_d_ptr = thrust::raw_pointer_cast(&dist_matrix_d[0]);

  float * dataset_d;
  cudaMalloc((void**) &dataset_d, n*m*sizeof(float));
  if (!dataset_d) {
    fprintf(stderr, " Cannot allocate cuda dataset %u array\n", n*n);
    exit(1);
  }

  cudaMemcpy(dataset_d, dataset, n*m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dataset_d, dataset, n*m*sizeof(float), cudaMemcpyHostToDevice);
  
  int thread_cnt = 1024;
  int block_cnt = (int) ceil((float)n*n / thread_cnt);
  calculate_pairwise_dists_cuda<<<block_cnt, thread_cnt>>>(dataset_d, dist_matrix_d_ptr, n, m);
  
  thrust::copy(dist_matrix_d.begin(), dist_matrix_d.end(), dist_matrix.begin());  
  // print_thrustfloat_matrix(dist_matrix, n, n);
  // std::cout <<"GPU: Calculated GPU Pairwise Dists" << std::endl;


  thrust::host_vector<unsigned int> indices(n*n);

  for(int i=0; i<n*n; i++){
    indices[i] = i;
  }

  thrust::device_vector<unsigned int> indices_d = indices;
     
  thrust::sort_by_key(dist_matrix_d.begin(), dist_matrix_d.end(), indices_d.begin());
   
  thrust::copy(indices_d.begin(), indices_d.end(), indices.begin());
  // std::cout <<"GPU: Sorted Pairwise Dists" << std::endl;

  // for(int i = 0; i < n*n; i++){
  //   int index = indices[i];
    
  //   int r = index/n, c = index%n;
  //   std::cout << r << " " << c << " :" << dist_matrix[index(r,c,n)] << std::endl;
  //   if (dist_matrix[index(r,c,n)] == FLT_MAX) break;
  // }

  // dist_matrix_d.clear();
  

  // thrust::device_vector<unsigned int> cluster_d(n);
  // cluster_d = cluster;
  // unsigned int * cluster_d_ptr = thrust::raw_pointer_cast(&cluster_d[0]);
  unsigned int * cluster_ptr = thrust::raw_pointer_cast(&cluster[0]);
  int iteration = 0, vec_idx_1, vec_idx_2;
  float distance;

  // std::cout <<"Building Dendrogram" << std::endl;
  for(int i=0; i<n*n; i++) {
    
    vec_idx_1 = indices[i]/n;
    vec_idx_2 = indices[i]%n;
    distance = dist_matrix[index(vec_idx_1, vec_idx_2, n)];
    
    if (distance == FLT_MAX) break;

    if (cluster[vec_idx_1] != cluster[vec_idx_2]){
        // std::cout << "Itr: "<< iteration << " " << std::endl;
        unsigned int cluster_1 = cluster[vec_idx_1];
        unsigned int cluster_2 = cluster[vec_idx_2];

        // CPU Update
        
        update_cluster_cpu(cluster_ptr, cluster_1, cluster_2, n);
        
        // // GPU Update
        // int thread_cnt = 1024;
        // int block_cnt = (int) ceil((float)n / thread_cnt);
        // update_cluster_cuda<<<block_cnt, thread_cnt>>>(cluster_d_ptr, cluster_1, cluster_2, n);
        // thrust::copy(cluster_d.begin(), cluster_d.end(), cluster.begin());
        
        
        // print_thrustint_matrix(cluster, 1, n);
        dendrogram[index(iteration, 0, 3)] = vec_idx_1;
        dendrogram[index(iteration, 1, 3)] = vec_idx_2;
        dendrogram[index(iteration, 2, 3)] = distance;
        iteration ++;
    }
  }
  // std::cout <<"Finished Building Dendrogram" << std::endl;
  cudaFree(dataset_d);

  if (PRINT_DENDRO)
    print_dendro(dendrogram, iteration);
}

void calculate_pairwise_dists(float * dataset, int n, int m, float * dist_matrix) {
  // O(n)
  for (int i = 0; i < n*n; i++) dist_matrix[i] = FLT_MAX;
  
  // O(n*n*m)
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // O(m)
      if (i==j) dist_matrix[index(i, j, n)] = FLT_MAX;
      else dist_matrix[index(i, j, n)] = calculate_dist(dataset, i, j, m);
    }
  }  
}

// passing vec1_i and vec2_i instead of float * as dist_matrix is 1-D
float calculate_dist(float * dataset, int i, int j, int dim) {
  float dist = 0;
  // O(m)
  for (int mi = 0; mi < dim; mi++){
    float x = (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]);
    dist += x * x;
  }
  return dist;
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

__global__ void update_cluster_cuda(unsigned int * cluster, unsigned int cluster_1, unsigned int cluster_2, unsigned int n)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  // Dont update if thread id is outside of the box
  if (index < n){
    // printf("Kernel: %d %d %d\n", index, cluster_1, cluster_2);
    if (cluster[index] == cluster_2) cluster[index] = cluster_1;
  }
}