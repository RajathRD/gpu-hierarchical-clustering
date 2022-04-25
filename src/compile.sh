#!/bin/bash
rm cpu1
rm cpu2
rm gpu1
rm gpu2
nvcc -o cpu1 sequential_clustering.cu
g++ -o cpu2 cpu_clustering.cpp
nvcc -o gpu1 single_gpu_clustering.cu
nvcc -o gpu2 single_gpu_clustering_sort.cu
