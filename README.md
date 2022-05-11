# RisGraph

## Description
This is the open-source implementation for RisGraph:

RisGraph: A Real-Time Streaming System for Evolving Graphs to Support Sub-millisecond Per-update Analysis at Millions Ops/s.  
Guanyu Feng, Zixuan Ma, Daixuan Li, Shengqi Chen, Xiaowei Zhu, Wentao Han, Wenguang Chen.  
SIGMOD/PODS '21: Proceedings of the 2021 International Conference on Management of Data. https://doi.org/10.1145/3448016.3457263

## System Dependency
 - [CMake](https://gitlab.kitware.com/cmake/cmake)
 - [TBB](https://github.com/oneapi-src/oneTBB) 
 - OpenMP and C++17
 - [Optional: UCX >= 1.8](https://github.com/openucx/ucx) 

## Compilation
```bash
git clone https://github.com/thu-pacman/RisGraph.git --recursive
cd RisGraph
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Preprocessing

The graph format used by RisGraph is binary edge lists with 64-bit vertex IDs (in host byte ordering).
RisGraph provides some tools that converts edge lists in text format (such as [SNAP](https://snap.stanford.edu/data) datasets) to binary format. 

RisGraph simulates sliding windows by inserting and deleting edges based on the sequence of edges from the input dataset.

### Sorting edges based on timestamps

```bash
# when edges are timestamped
# each line of text_graph_path is an edge with three integers
# source_vertex_id destination_vertex_id edge_timestamp

./convert_to_binary_timestamp < text_graph_path > binary_graph_path
```

### Randomly shuffling edges

```bash
# when edges have no specific order (for most of public datasets)
# each line of text_graph_path is an edge with two integers
# source_vertex_id destination_vertex_id

./convert_to_binary_random < text_graph_path > binary_graph_path
```

### Keeping the input ordering

```bash
# when edges are already in chronological or custom order
# each line of text_graph_path is an edge with two integers
# source_vertex_id destination_vertex_id

./convert_to_binary < text_graph_path > binary_graph_path
```

## Entire Graph Processing
These applications will process the entire graph. 

### Breadth-First Search
```bash
./bfs binary_graph_path root
```

### Single Source Shortest Path
```bash
./sssp binary_graph_path root
```

### Single Source Widest Path
```bash
./sswp binary_graph_path root
```

### Weakly Connected Components
```bash
# edges are treated as undirected edges
./wcc binary_graph_path
```

## Incremental Processing
These applications will load the first `initial_edges_percent` edges from the graph, insert and delete an edge as an update (simulating sliding windows), and incrementally process the algorithm.

### Breadth-First Search
```bash
./bfs_inc binary_graph_path root initial_edges_percent
```

### Single Source Shortest Path
```bash
./sssp_inc binary_graph_path root initial_edges_percent
```

### Single Source Widest Path
```bash
./sswp_inc binary_graph_path root initial_edges_percent
```

### Weakly Connected Components
```bash
./wcc_inc binary_graph_path initial_edges_percent
```

## Incremental Processing with Safe/Unsafe Classification and Latency-aware Scheduler
These applications will load the first `initial_edges_percent` edges from the graph and simulate `num_of_clients` clients requesting updates with inserting/deleting an edge (simulating sliding windows). 

RisGraph enables safe/unsafe classification for incrementally processing and tries to make `tail_latency_percent` updates are within `target_tail_latency` milliseconds through the latency-aware scheduler. It is recommended to set `num_of_clients` starting with the number of physical cores and try doubling `num_of_clients` until RisGraph cannot fulfill the expected tail latency.

### Breadth-First Search
```bash
./bfs_inc_rt binary_graph_path root initial_edges_percent target_tail_latency tail_latency_percent num_of_clients
```

### Single Source Shortest Path
```bash
./sssp_inc_rt binary_graph_path root initial_edges_percent target_tail_latency tail_latency_percent num_of_clients
```

### Single Source Widest Path
```bash
./sswp_inc_rt binary_graph_path root initial_edges_percent target_tail_latency tail_latency_percent num_of_clients
```

### Weakly Connected Components
```bash
./wcc_inc_rt binary_graph_path initial_edges_percent target_tail_latency tail_latency_percent num_of_clients
```


## Incremental Processing with Batched Updates
These applications will load the first `initial_edges_percent` edges from the graph, insert and delete `batch_size` edges as a batched update (simulating sliding windows), and incrementally process the algorithm.

### Breadth-First Search
```bash
./bfs_inc_batch binary_graph_path root initial_edges_percent batch_size
```

### Single Source Shortest Path
```bash
./sssp_inc_batch binary_graph_path root initial_edges_percent batch_size
```

### Single Source Widest Path
```bash
./sswp_inc_batch binary_graph_path root initial_edges_percent batch_size
```

### Weakly Connected Components
```bash
./wcc_inc_batch binary_graph_path initial_edges_percent batch_size
```


## Performance Tuning
* Binding threads to CPU cores: `export OMP_PROC_BIND=true`
* Trying different NUMA policies: In our experiments, `numactl --preferred=0` achieves relatively good performance in a wide range of cases.
