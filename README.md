# Accelerating Hierarchical Clustering with GPU

#### Sanjar Ahmadov, Rajath Ravindranath Dani, & Ameya Shere

### Directory Structure

This codebase is structured as follows.
- `src` folder contains the source code for all 4 implementations
- `target` folder contains the binaries for all 4 implementations
- `compile.sh` compiles the source code and generates the binaries
- `run_experiments.py` runs experiments for different problem sizes across the different clusters
- `collect_experiments_results.py` collects and generates a summary of the experiments results in a given folder

```
.
├── compile.sh
├── run_experiments.py
├── collect_experiments_results.py
└── src
│   ├── cpu_clustering.cpp
│   ├── naive_cpu_clustering.cpp
│   ├── single_gpu_clustering.cu
│   └── single_gpu_clustering_sort.cu
└── target
    ├── cpu1
    ├── cpu2
    ├── gpu1
    └── gpu2
```

### Instructions
1. `cd` to root
2. Enable script permissions: `chmod +x compile.sh`
3. Compile and generate binaries in `target` folder: `./compile.sh`
4. To run any of the 4 binaries: `./target/<binary-name> N M`, where `N` is number of samples to generate and `M` is dimension of samples
5. To run the experiments: `./run_experiments.py folder_path`
6. To get the outputs of finished experiments: `./collect_experiments_results.py folder_path`
