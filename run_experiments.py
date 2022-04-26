#!/usr/bin/env python
import os
import sys

def read_exp_res(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Time"):
                runtime_str = line.split(":")[1]
                seconds = float(runtime_str)
                return seconds
        return "N/A"


def doesResultExist(file_path):
    return os.path.isfile(file_path) and read_exp_res(file_path) != "N/A"

def run_experiments_parallel(build_name, ns, ms, experiments_folder):
    results = []
    timeout_seconds = 5*3600
    build = os.path.join(experiments_folder, build_name)
    for n in ns:
        for m in ms:
            title = build_name + "_" + str(n) + "_" + str(m) + ".txt"
            result_file = os.path.join(experiments_folder, title)
            if doesResultExist(result_file):
                print(result_file + " exists, skipping running experiment")
                continue
            command = ("timeout " + str(timeout_seconds) +" ./" + build + " " + str(n) 
                + " " + str(m) + " > " + result_file + " 2>&1 &")
            print("Running: " + command)
            os.system(command)
    return results


def run_experiments_sequential(build_name, ns, ms, experiments_folder):
    results = []
    timeout_seconds = 3600
    build = os.path.join(experiments_folder, build_name)

    # Create empty files initially for reporting
    for n in ns:
        for m in ms:
            title = build_name + "_" + str(n) + "_" + str(m) + ".txt"
            result_file = os.path.join(experiments_folder, title)
            if doesResultExist(result_file):
                print(result_file + " exists, skipping creating it")
                continue
            command = ("touch " + result_file)
            os.system(command)

    # Run experiments sequentially
    for n in ns:
        for m in ms:
            title = build_name + "_" + str(n) + "_" + str(m) + ".txt"
            result_file = os.path.join(experiments_folder, title)
            if doesResultExist(result_file):
                print(result_file + " exists, skipping running experiment")
                continue
            command = ("timeout " + str(timeout_seconds) +" ./" + build + " " + str(n) 
                + " " + str(m) + " > " + result_file + " 2>&1")
            print("Running: " + command)
            os.system(command)
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: program experiments_folder")
        exit(1)

    experiments_folder = sys.argv[1]
    #os.system("rm -rf " + experiments_folder)
    #os.system("mkdir -p " + experiments_folder)

    os.system("g++ -o " + os.path.join(experiments_folder, "cpu1") + " " + os.path.join("src", "naive_cpu_clustering.cpp"))
    os.system("g++ -o " + os.path.join(experiments_folder, "cpu2") + " " + os.path.join("src", "cpu_clustering.cpp"))
    os.system("nvcc -o " + os.path.join(experiments_folder, "gpu1") + " " + os.path.join("src", "single_gpu_clustering.cu"))
    os.system("nvcc -o " + os.path.join(experiments_folder, "gpu2") + " " + os.path.join("src", "single_gpu_clustering_sort.cu"))

    n = [4096, 8192, 12288, 16384]
    m = [16, 32, 64, 128, 1024, 2048, 4096, 8192]

    run_experiments_parallel("cpu1", n, m, experiments_folder)
    run_experiments_parallel("cpu2", n, m, experiments_folder)
    run_experiments_sequential("gpu1", n, m, experiments_folder)
    run_experiments_sequential("gpu2", n, m, experiments_folder)

main()