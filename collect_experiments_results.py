#!/usr/bin/env python
import os

def print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu_runtimes):
    if  not (len(cpu1_runtimes) == len(cpu2_runtimes) and 
        len(cpu2_runtimes) == len(gpu_runtimes)):
        raise Exception("All lengths should match")
    
    print("i\t\tn\t\tm\t\tCPU1\t\tCPU2\t\tGPU\t\t(CPU1/GPU)\t\t(CPU2/GPU)")
    l = len(cpu1_runtimes)
    for i in range(l):
        cpu1_entry = cpu1_runtimes[i]
        cpu2_entry = cpu2_runtimes[i]
        gpu2_entry = gpu_runtimes[i]
        n = cpu1_entry[0]
        m = cpu1_entry[1]

        cpu1 = "{:.2f}".format(cpu1_entry[2]) if cpu1_entry[2] != "N/A" else "N/A"
        cpu2 = "{:.2f}".format(cpu2_entry[2]) if cpu2_entry[2] != "N/A" else "N/A"
        gpu2 = "{:.2f}".format(gpu2_entry[2]) if gpu2_entry[2] != "N/A" else "N/A"
        sp1 = "{:.2f}".format(cpu1_entry[2]/gpu2_entry[2]) if cpu1_entry[2] != "N/A" and gpu2_entry[2] != "N/A" else "N/A"
        sp2 = "{:.2f}".format(cpu2_entry[2]/gpu2_entry[2]) if cpu2_entry[2] != "N/A" and gpu2_entry[2] != "N/A" else "N/A"

        print(str(i)+"\t\t"+str(n)+"\t\t"+str(m)+"\t\t"+cpu1+"\t\t"+cpu2+"\t\t"+gpu2+"\t\t"+sp1+"\t\t"+sp2)


def read_exp_res(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Time"):
                runtime_str = line.split(":")[1]
                seconds = float(runtime_str)
                return seconds
        return "N/A"

def collect_results(experiments_folder):
    cpu1_runtimes = []
    cpu2_runtimes = []
    gpu1_runtimes = []
    gpu2_runtimes = []
    all_filenames = sorted([f for f in os.listdir(experiments_folder) if os.path.isfile(os.path.join(experiments_folder, f))])
    for file_name in all_filenames:
        file_name_split = file_name.split("_")
        n = int(file_name_split[1])
        m = int(file_name_split[2].split(".")[0])
        file_path = os.path.join(experiments_folder, file_name)

        entry = (n, m, read_exp_res(file_path))
        if file_name.startswith("cpu1"):
            cpu1_runtimes.append(entry)
        elif file_name.startswith("cpu2"):
            cpu2_runtimes.append(entry)
        elif file_name.startswith("gpu1"):
            gpu1_runtimes.append(entry)
        elif file_name.startswith("gpu2"):
            gpu2_runtimes.append(entry)
        else:
            raise Exception("Unknown file")
    return cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes

def main():
    experiments_folder = "experiments_cuda1"
    cpu1_runtimes, cpu2_runtimes, _, gpu_runtimes = collect_results(experiments_folder)
    print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu_runtimes)

main()