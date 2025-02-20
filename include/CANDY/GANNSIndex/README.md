# GANNS
## Introduction
This project includes (1) a GPU-based algorithm GANNS which can accelerate 
the ANN search on proximity graphs by re-designing the classical CPU-based search algorithm 
and using GPU-friendly data structures. 
(2) novel GPU-based proximity graph construction algorithms which ensure the quality of the resulting proximity graph.

## Usage
Step 1. Generate template
```zsh
./generate_template.sh
```

### Search
To use search algorithm, generate query instance.
```zsh
./generate_query_instances.sh [dim] [metric]
```
For instance, the following command line generates an executable program which can work on datasets with dimension 128 and metric euclidean distance. 
```zsh
./generate_query_instances.sh 128 l2
```
Currently, we support dimension ```no larger than 960``` and three metrics: ```euclidean distance (l2)```, ```cosine similarity (cos)``` and ```inner product (ip)```.

To use the generated executable program, the following parameters need to be provided.
```zsh
./query_128_l2 [base_path] [query_path] [graph_type] [graph_path] [groundtruth_path] [e] [k]
```
Specifically, ```[base_path]``` is the directory of data points (database); ```[query_path]``` is the directory of query points; ```[graph_type]``` is the type of proximity graph; 
```[groundtruth_path]``` is the directory of groundtruth; ```[e]``` represents the number of explored vertices; ```[k]``` denotes the number of returned nearest neighbors;

For instance, the following command line performs ANN search on NSW constructed on SIFT dataset 
(These files in the command line are not included in this project due to their size).
```zsh
./query_128_l2 ../dataset/sift/base.fvecs ../dataset/sift/query.fvecs nsw ../dataset/sift/base.fvecs_64_16.nsw ../dataset/sift/groundtruth.ivecs 64 10
```
Currently, we support two proximity graphs: ```NSW (nsw)``` and ```HNSW (hnsw)```.

### Construction
To use construction algorithm, generate build instance.
```zsh
./generate_build_instances.sh [dim] [metric]
```
Similarly, we support dimension ```no larger than 960``` and three metrics: ```euclidean distance (l2)```, ```cosine similarity (cos)``` and ```inner product (ip)```.

To use the generated executable program, the following parameters need to be provided.
```zsh
./build_128_l2 [base_path] [graph_type] [e] [d_min]
```
Specifically, ```[base_path]``` is the directory of data points (database); ```[graph_type]``` is the type of proximity graph; 
```[e]``` represents the number of explored vertices; ```[d_min]``` denotes minimum degree in the proximity graph (by default, d_max = 2 * d_min);

For instance, the following command line establishes a HNSW graph on SIFT dataset 
(These files in the command line are not included in this project due to their size).
```zsh
./build_128_l2 ../dataset/sift/base.fvecs hnsw 64 16
```
Similarly, we support two proximity graphs: ```NSW (nsw)``` and ```HNSW (hnsw)```.

Notice the parameters ```dim``` and ```metric``` that are provided to generate query (resp. build) instances must be consistent with dimension and metric of datasets. 
Otherwise, the executable program may shut down, or the recall (resp. the quality of proximity graph) may be poor.

## Datasets
The information of used real-world datasets is provided in our paper. Currently, we support ```.fvecs``` file for base datasets and query datasets 
and ```.ivecs``` for groundtruth file.
