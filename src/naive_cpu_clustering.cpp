#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
#include <math.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/* Config params */
#define PRINT_LOG 1
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
  // The CPU sequential version 
  start = clock();
  seq_clustering(dataset, n, m, dendrogram);    
  end = clock();
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

  if (PRINT_LOG) {
    printf("Dendrogram:\n");
    print_float_matrix(dendrogram, n-1, 3);
  }
  printf("Time taken for CPU %lf\n", time_taken);
  
  free(dataset);

  return 0;
}


/*****************  The CPU sequential version **************/
void  seq_clustering(float * dataset, unsigned int n, unsigned int m, float * dendrogram)
{
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  int * result = (int *)calloc(n, sizeof(int));
  if( !result ) {
   fprintf(stderr, " Cannot allocate result %u array\n", n);
   exit(1);
  }

  for (int i = 0; i < n; i++) result[i] = i;

  float* dist_matrix = (float *)calloc(n*n, sizeof(float));
  if( !dist_matrix ) {
   fprintf(stderr, " Cannot allocate dist_matrix %u array\n", n*n);
   exit(1);
  }

  // O(n*n*m) -> GPU
  start = clock();
  calculate_pairwise_dists(dataset, n, m, dist_matrix);
  end = clock();

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  // if (PRINT_ANALYSIS)
  //   printf("Time taken for distance computation: %lf\n", time_taken);
  
  start = clock();
  for (int iteration=0; iteration < n - 1; iteration++) {
    float entry[3]; 

    // O(I*n*n) -> GPU
    find_pairwise_min(dist_matrix, n, entry, result);
    
    dendrogram[index(iteration, 0, 3)] = entry[0];
    dendrogram[index(iteration, 1, 3)] = entry[1];
    dendrogram[index(iteration, 2, 3)] = entry[2];

    // O(I*n) -> amortized O(I)
    merge_clusters(result, (int)entry[0], (int)entry[1], n);
  }

  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  if (PRINT_ANALYSIS)
      printf("Time taken for merge cluster, Iteration %lf\n",time_taken);
    
  for (int i=0; i<n; i++) result[i] = get_parent(i, result);

  free(dist_matrix);
  free(result);
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
  for (int mi = 0; mi < dim; mi++) {
    float x = (dataset[index(i, mi, dim)] - dataset[index(j,mi,dim)]);
    dist += x * x;
  }
  return dist;
}

int get_parent(int curr_parent, int* parents) {
  if (parents[curr_parent] == curr_parent) return curr_parent;
  parents[curr_parent] = get_parent(parents[curr_parent], parents);
  return parents[curr_parent];
}


void find_pairwise_min(float * dist_matrix, int n, float* entry, int* parents) {
  entry[0] = 0;
  entry[1] = 0;
  entry[2] = FLT_MAX;
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      if (get_parent(i, parents) != get_parent(j, parents)) {
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
  int parent_i = get_parent(data_point_i, result);
  result[get_parent(data_point_j, result)] = parent_i;
}