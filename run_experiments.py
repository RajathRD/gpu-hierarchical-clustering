#!/usr/bin/env python
import os

def isNotStr(a):
    return type(a) != type("")

def print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes):
    if  not (len(cpu1_runtimes) == len(cpu2_runtimes) and 
        len(cpu2_runtimes) == len(gpu1_runtimes) and
        len(gpu1_runtimes) == len(gpu2_runtimes)):
        raise Exception("All lengths should match")
    
    print("i\tn\tm\tCPU1\tCPU2\tGPU1\tGPU2\t(CPU1/GPU2)\t(CPU2/GPU2)\t(GPU1/GPU2)")
    l = len(cpu1_runtimes)
    for i in range(l):
        cpu1 = cpu1_runtimes[i]
        cpu2 = cpu2_runtimes[i]
        gpu1 = gpu1_runtimes[i]
        gpu2 = gpu2_runtimes[i]
        n = cpu1[0]
        m = cpu1[1]
        sp1 = sp2 = sp3 = "Error"
        if isNotStr(cpu1[2]) and isNotStr(gpu2[2]):
            sp1 = "{:.4f}".format(cpu1[2]/gpu2[2]) 
        if isNotStr(cpu2[2]) and isNotStr(gpu2[2]):
            sp2 = "{:.4f}".format(cpu2[2]/gpu2[2])
        if isNotStr(gpu1[2]) and isNotStr(gpu2[2]):
            sp3 = "{:.4f}".format(gpu1[2]/gpu2[2])

        print(str(i)+"\t"+str(n)+"\t"+str(m)+"\t"+str(cpu1[2])+"\t"+str(cpu2[2])+"\t"+
            str(gpu1[2])+"\t"+str(gpu2[2])+"\t"+sp1+"\t\t"+sp2+"\t\t"+sp3)


def read_exp_res(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Time"):
                runtime_str = line.split(":")[1]
                seconds = float(runtime_str)
                return seconds
        return "Error"

def run_experiments(build_name, ns, ms, build_folder, experiments_folder):
    results = []
    timeout_seconds = 10
    build = os.path.join(build_folder, build_name)
    for n in ns:
        for m in ms:
            title = build_name + "_" + str(n) + "_" + str(m) + ".txt"
            result_file = os.path.join(experiments_folder, title)
            command = "timeout " + str(timeout_seconds) +" ./" + build + " " + str(n) + " " + str(m) + " > " + result_file + " 2>&1 "
            #command = "(timeout " + str(timeout_seconds) + " ./" + build + " " + str(n) + " " + str(m) + " > /dev/null 2>&1)  2> " + result_file 
            print("Running: " + command)
            os.system(command)
            results.append((n, m, read_exp_res(result_file)))
    return results

def main():
    build_folder = "target"
    os.system("./compile.sh")

    experiments_folder = "experiments"
    os.system("rm -rf " + experiments_folder)
    os.system("mkdir -p " + experiments_folder)

    #n = [4096, 8192, 12288, 16384]
    #m = [16, 32, 64, 128, 1024, 2048, 4096]

    n = [1000, 2000]
    m = [16, 32]

    cpu1_runtimes = run_experiments("cpu1", n, m, build_folder, experiments_folder)
    cpu2_runtimes = run_experiments("cpu2", n, m, build_folder, experiments_folder)
    gpu1_runtimes = run_experiments("gpu1", n, m, build_folder, experiments_folder)
    gpu2_runtimes = run_experiments("gpu2", n, m, build_folder, experiments_folder)

    print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes)

main()