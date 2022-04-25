#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
#include <math.h>

__global__ void min_reduction(float *, unsigned int *, unsigned int *, unsigned int n);

int main(){
    unsigned int n = 1024*1024;
    unsigned int size = n;
    int thread_cnt = 256, block_cnt = (int)ceil((float)n/thread_cnt);
    
    float * arr, * arr_d;
    unsigned int * indices, * indices_d;
    unsigned int * block_mins, * block_mins_d;
    block_mins = (unsigned int *) calloc (size, sizeof(unsigned int));
    
    arr = (float *) calloc (n, sizeof(float));
    indices = (unsigned int *) calloc (n, sizeof(unsigned int));
    unsigned int * data = &indices[0];

    for (int i=0; i<n; i++){
        arr[i] = i*1.0;
        indices[i] = i;
        // printf("%d -> %.1f ; ", indices[i], arr[i]);
    }
    // printf("\n");

    cudaMalloc((void**) &block_mins_d, n*sizeof(unsigned int));
    cudaMalloc((void**) &arr_d, n*sizeof(float));
    cudaMalloc((void**) &indices_d, n*sizeof(unsigned int));
    
    cudaMemcpy(arr_d, arr, n*sizeof(float), cudaMemcpyHostToDevice);

    while(size > 1){
        block_cnt = (int)ceil((float)size/thread_cnt);
        printf("Num Blocks = %d; Num Threads = %d\n", block_cnt, thread_cnt);
        
        cudaMemcpy(indices_d, data, size*sizeof(unsigned int), cudaMemcpyHostToDevice);
        min_reduction<<<block_cnt, thread_cnt>>>(arr_d, indices_d, block_mins_d, size);
        cudaMemcpy(block_mins, block_mins_d, block_cnt*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        printf("Block Mins: ");
        for(int i=0; i<block_cnt; i++) printf("%d ", block_mins[i]);
        printf("\n");

        data = block_mins;
        size = block_cnt;
    }

    printf("Max Val: %d", block_mins[0]);

}

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
        //   printf("stride: %d, left: %d, right: %d\n", stride, left, right);
          // printf("a[%d]=%.2f, a[%d]=%.2f\n", indices[left], arr[indices[left]], indices[right], arr[indices[right]]);
          if(arr[indices[left]] < arr[indices[right]]){
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