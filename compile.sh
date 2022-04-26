#!/bin/bash
rm target/cpu1
rm target/cpu2
rm target/gpu1
rm target/gpu2
g++ -o target/cpu1 src/naive_cpu_clustering.cpp
g++ -o target/cpu2 src/cpu_clustering.cpp
nvcc -o target/gpu1 src/single_gpu_clustering.cu
nvcc -o target/gpu2 src/single_gpu_clustering_sort.cu
