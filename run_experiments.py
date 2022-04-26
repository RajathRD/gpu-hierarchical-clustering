#!/usr/bin/env python
import os
import sys

def run_experiments(build_name, ns, ms, experiments_folder, is_parallel):
    results = []
    timeout_seconds = 3600
    build = os.path.join(experiments_folder, build_name)
    for n in ns:
        for m in ms:
            title = build_name + "_" + str(n) + "_" + str(m) + ".txt"
            result_file = os.path.join(experiments_folder, title)
            background = "&" if is_parallel else ""
            command = ("timeout " + str(timeout_seconds) +" ./" + build + " " + str(n) 
                + " " + str(m) + " > " + result_file + " 2>&1 " + background)
            print("Running: " + command)
            os.system(command)
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: program experiments_folder")
        exit(1)

    experiments_folder = sys.argv[1]
    os.system("rm -rf " + experiments_folder)
    os.system("mkdir -p " + experiments_folder)

    "g++ -o " + os.path.join(experiments_folder, "cpu1") + " " + os.path.join("src", "naive_cpu_clustering.cpp")
    "g++ -o " + os.path.join(experiments_folder, "cpu2") + " " + os.path.join("src", "cpu_clustering.cpp")
    "nvcc -o " + os.path.join(experiments_folder, "gpu1") + " " + os.path.join("src", "single_gpu_clustering.cu")
    "nvcc -o " + os.path.join(experiments_folder, "gpu2") + " " + os.path.join("src", "single_gpu_clustering_sort.cu")

    n = [200, 1000]
    m = [10, 20]

    # n = [4096, 8192, 12288, 16384]
    # m = [16, 32, 64, 128, 1024, 2048, 4096]

    run_experiments("cpu1", n, m, experiments_folder, True)
    run_experiments("cpu2", n, m, experiments_folder, True)
    #run_experiments("gpu1", n, m, experiments_folder, False)
    run_experiments("gpu2", n, m, experiments_folder, False)

main()