CXX=g++
NVCC=nvcc
DISTTYPE=USE_L2_DIST_
FLAG_DEBUG=-O3

all : build query

build : build.cu data.h graph_index/navigable_small_world.h graph_index/nsw_graph_operations.h graph_index/hierarchical_navigable_small_world.h
	$(NVCC) -ccbin g++ -I../../Common  -rdc=true -m64 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_75,code=compute_75 \
	-std=c++11 build.cu $(FLAG_DEBUG) -o build -Xptxas -v \
	-D$(DISTTYPE)

query : query.cu data.h graph_index/navigable_small_world.h graph_index/nsw_graph_operations.h graph_index/hierarchical_navigable_small_world.h
	$(NVCC) -ccbin g++ -I../../Common  -rdc=true -m64 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_75,code=compute_75 \
	-std=c++11 query.cu $(FLAG_DEBUG) -o query -Xptxas -v \
	-D$(DISTTYPE)