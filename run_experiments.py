#!/usr/bin/env python
import os
import sys

def run_experiments(build_name, ns, ms, build_folder, experiments_folder, is_parallel):
    results = []
    timeout_seconds = 3600
    build = os.path.join(build_folder, build_name)
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

    build_folder = "target"
    os.system("./compile.sh")

    experiments_folder = sys.argv[1]
    os.system("rm -rf " + experiments_folder)
    os.system("mkdir -p " + experiments_folder)

    # n = [4096, 8192, 12288, 16384]
    # m = [16, 32, 64, 128, 1024, 2048, 4096]

    n = [200, 1000]
    m = [10, 20]

    run_experiments("cpu1", n, m, build_folder, experiments_folder, True)
    run_experiments("cpu2", n, m, build_folder, experiments_folder, True)
    #run_experiments("gpu1", n, m, build_folder, experiments_folder, False)
    run_experiments("gpu2", n, m, build_folder, experiments_folder, False)

main()