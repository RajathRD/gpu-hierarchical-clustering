# Accelerating Hierarchical Clustering with GPU

#### Sanjar Ahmadov, Rajath Ravindranath Dani, & Ameya Shere

### Directory Structure

This codebase is structured as follows.
- `src` folder contains the source code for all 4 implementations
- `target` folder contains the binaries for all 4 implementations
- `compile.sh` compiles the source code and generates the binaries
- `run_experiments.py` runs experiments for different problem sizes across the different clusters
.
├── compile.sh
├── run_experiments.py
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

### Instructions
1. `cd` to root
2. Enable script permissions: `chmod +x compile.sh`
3. Compile and generate binaries in targets folder: `./compile.sh`
4. To run any of the 4 binaries: `./targets/<binary-name> N M`, where `N` is number of samples to generate and `M` is dimension of samples
5. To run the experiments: `python run_experiments.py`
