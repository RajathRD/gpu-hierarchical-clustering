#!/bin/bash
rm cpu1
rm cpu2
rm gpu1
rm gpu2
gcc -o cpu1 naive_cpu_clustering.c
g++ -o cpu2 cpu_clustering.cpp
nvcc -o gpu1 single_gpu_clustering.cu
nvcc -o gpu2 single_gpu_clustering_sort.cu
