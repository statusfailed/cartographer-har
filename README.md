# HAR

This is an implementation of HARs: a datastructure (and algorithms) for
implementing string diagrams using sparse adjacency matrices.

It accompanies [this paper](https://arxiv.org/abs/2105.09257)

For the **sparse-matrix-based implementation**, see the file
[cartographer_har/sparse_slice.py](cartographer_har/sparse_slice.py)

For **HAR benchmarks**, see the [benchmarks](./benchmarks) folder

For the corresponding **catlab benchmarks**, see
[catlab-benchmark.jl](./julia/catlab-benchmark.jl)

# Raw data of experiments

See the [./results](./results) folder.
**NOTE CAREFULLY**: the data in `catlab-results.csv` uses time in nanoseconds,
while the data in `har-results.csv` has time in seconds.
