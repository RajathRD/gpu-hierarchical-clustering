#!/usr/bin/env python
import os
import sys 
import pandas

def print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes):
    if  not (len(cpu1_runtimes) == len(cpu2_runtimes) and 
        len(cpu2_runtimes) == len(gpu1_runtimes) and
        len(gpu1_runtimes) == len(gpu2_runtimes)):
        raise Exception("All lengths should match")
    
    # print("i\t\tn\t\tm\t\tCPU1\t\tCPU2\t\tGPU\t\t(CPU1/GPU)\t(CPU2/GPU)")
    headers = ["n", "m", "CPU1", "CPU2", "GPU1", "GPU2", "(CPU1/GPU1)", "(CPU2/GPU1)", "(CPU1/GPU2)", "(CPU2/GPU2)", "(GPU1/GPU2)"]
    df = pandas.DataFrame(columns=headers)
    l = len(cpu1_runtimes)
    for i in range(l):
        cpu1_entry = cpu1_runtimes[i]
        cpu2_entry = cpu2_runtimes[i]
        gpu1_entry = gpu1_runtimes[i]
        gpu2_entry = gpu2_runtimes[i]
        n = cpu1_entry[0]
        m = cpu1_entry[1]

        if not(cpu1_entry[0] == cpu2_entry[0] and cpu2_entry[0] == gpu2_entry[0] and cpu1_entry[1] == cpu2_entry[1] and cpu2_entry[1] == gpu2_entry[1]):
            raise Exception("Runtimes have not been passed in correct order in terms of inputs")

        cpu1 = cpu1_entry[2] if cpu1_entry[2] != "N/A" else "N/A"
        cpu2 = cpu2_entry[2] if cpu2_entry[2] != "N/A" else "N/A"
        gpu1 = gpu1_entry[2] if gpu1_entry[2] != "N/A" else "N/A"
        gpu2 = gpu2_entry[2] if gpu2_entry[2] != "N/A" else "N/A"
        sp1 = cpu1_entry[2]/gpu1_entry[2] if cpu1_entry[2] != "N/A" and gpu1_entry[2] != "N/A" else "N/A"
        sp2 = cpu2_entry[2]/gpu1_entry[2] if cpu2_entry[2] != "N/A" and gpu1_entry[2] != "N/A" else "N/A"
        sp3 = cpu1_entry[2]/gpu2_entry[2] if cpu1_entry[2] != "N/A" and gpu2_entry[2] != "N/A" else "N/A"
        sp4 = cpu2_entry[2]/gpu2_entry[2] if cpu2_entry[2] != "N/A" and gpu2_entry[2] != "N/A" else "N/A"
        sp5 = gpu1_entry[2]/gpu2_entry[2] if gpu1_entry[2] != "N/A" and gpu2_entry[2] != "N/A" else "N/A"
        
        entry = (n, m, cpu1, cpu2, gpu1, gpu2, sp1, sp2, sp3, sp4, sp5)
        # print(entry)
        df.loc[i] = entry

        # print(str(i)+"\t\t"+str(n)+"\t\t"+str(m)+"\t\t"+cpu1+"\t\t"+cpu2+"\t\t"+gpu2+"\t\t"+sp1+"\t\t"+sp2)

    df = df.sort_values(by=['n', "m"])
    print(df)

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
    all_filenames = sorted([f for f in os.listdir(experiments_folder) if (os.path.isfile(os.path.join(experiments_folder, f)) and f.endswith(".txt"))])
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
    if len(sys.argv) != 2:
        print("Usage: program experiments_folder")
        exit(1)

    experiments_folder = sys.argv[1]
    if not os.path.isdir(experiments_folder):
        print("Experiment folder " + experiments_folder + " does not exist!")
        exit(1)

    cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes = collect_results(experiments_folder)
    print_runtimes(cpu1_runtimes, cpu2_runtimes, gpu1_runtimes, gpu2_runtimes)

main()