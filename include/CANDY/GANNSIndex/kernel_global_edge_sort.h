#pragma once

#include "structure_on_device.cuh"
namespace cg = cooperative_groups;

__global__ 
void GlobalEdgesSort(KernelPair<float, int>* neighbors, Edge* edge_list, int* flags, int num_of_edges, int num_of_valid_edges, int total_num_of_points) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int t_id_global = b_id * blockDim.x + t_id;

    cg::grid_group grid = cg::this_grid();

    if(t_id_global >= num_of_valid_edges){
        edge_list[t_id_global].source_point = total_num_of_points;
        edge_list[t_id_global].target_point = total_num_of_points;
        edge_list[t_id_global].distance = Max;
    }
    cg::sync(grid);

    int step_id;
    int substep_id;

    //bitonic sort for all edges
    step_id = 1;
    substep_id = 1;
    Edge temparory_edge;
        
    for (; step_id <= num_of_edges / 2; step_id *= 2) {
        substep_id = step_id;
        
        for (; substep_id >= 1; substep_id /= 2) {
    
            int mastered_tid = (t_id_global / substep_id) * 2 * substep_id + (t_id_global & (substep_id - 1));

            if (mastered_tid < num_of_edges) {
                if ((t_id_global / step_id) % 2 == 0) {
                    if (edge_list[mastered_tid].target_point > edge_list[mastered_tid + substep_id].target_point||
                        (edge_list[mastered_tid].target_point == edge_list[mastered_tid + substep_id].target_point&&
                            edge_list[mastered_tid].distance > edge_list[mastered_tid + substep_id].distance)) {

                        temparory_edge = edge_list[mastered_tid];
                        edge_list[mastered_tid] = edge_list[mastered_tid + substep_id];
                        edge_list[mastered_tid + substep_id] = temparory_edge;

                    }
                } else {
                    if (edge_list[mastered_tid].target_point < edge_list[mastered_tid + substep_id].target_point||
                        (edge_list[mastered_tid].target_point == edge_list[mastered_tid + substep_id].target_point&&
                            edge_list[mastered_tid].distance < edge_list[mastered_tid + substep_id].distance)) {

                        temparory_edge = edge_list[mastered_tid];
                        edge_list[mastered_tid] = edge_list[mastered_tid + substep_id];
                        edge_list[mastered_tid + substep_id] = temparory_edge;

                    }
                }
            }

            cg::sync(grid);
                    
        }

    }

    int crt_id = edge_list[t_id_global].target_point;
    int last_id;

    if (t_id_global != 0) {
        last_id = edge_list[t_id_global - 1].target_point;
    }

    if (crt_id == last_id) {
        flags[t_id_global] = 0;
    } else {
        flags[t_id_global] = 1;
    }

    if (t_id_global == 0) {
        flags[t_id_global] = 1;
    }

    int flag_of_crt_edge = flags[t_id_global];

    int offset = 1;
    
    for (int d = num_of_edges>>1; d > 0; d >>= 1) {
        cg::sync(grid);

        if (t_id_global < d) { 
            int ai = offset * (2 * t_id_global + 1) - 1;
            int bi = offset * (2 * t_id_global + 2) - 1;
            flags[bi] += flags[ai];
        }
            
        offset *= 2; 
    }
    
    if (t_id_global == 0) { flags[num_of_edges - 1] = 0; } // clear the last element
    
    for (int d = 1; d < num_of_edges; d *= 2) {      
        offset >>= 1;
        cg::sync(grid);
        
        if (t_id_global < d) { 
            int ai = offset * (2 * t_id_global + 1) - 1;     
            int bi = offset * (2 * t_id_global + 2) - 1;
            int t = flags[ai]; 
            flags[ai] = flags[bi];
            flags[bi] += t;       
        } 
    }

    cg::sync(grid);

    int temparory_flag = flags[t_id_global];
    temparory_flag += flag_of_crt_edge;
    flags[t_id_global] = temparory_flag;

    int temparory_num_of_edges = flags[num_of_edges - 1];

    if (t_id_global == 0) { flags[num_of_edges] = temparory_num_of_edges + 1; }

    int flag_of_last_edge;

    cg::sync(grid);
                
    if (t_id_global > 0) {
        flag_of_last_edge = flags[t_id_global - 1];
    }

    cg::sync(grid);

    if (temparory_flag == flag_of_last_edge + 1) {
        flags[temparory_flag - 1] = t_id_global;
    }

    cg::sync(grid);

    if (t_id_global == 0) {
        flags[0] = 0;
    }

    if (t_id_global >= temparory_num_of_edges) {
        flags[t_id_global] = num_of_valid_edges;
    }

    cg::sync(grid);
    
}

/*extern "C" __global__ void SortSubSeqonGpu(Edge* edgeGlobalDev, int num_of_edges, int total_num_of_points){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int seqLen = blockDim.x;

    extern __shared__ Edge eMem[];
    Edge* edge_list = (Edge*)eMem;

    if(b_id*seqLen+t_id < num_of_edges){
        edge_list[t_id] = edgeGlobalDev[b_id*seqLen+t_id];
    }else{
        edge_list[t_id].source_point = total_num_of_points;
        edge_list[t_id].target_point = total_num_of_points;
        edge_list[t_id].source_pointIndex = -1;
        edge_list[t_id].target_pointIndex = -1;
        edge_list[t_id].distance = Max;
    }

    __syncthreads();

    int step_id = 1;
    int substep_id = 1;
    Edge temparory_edge;
        
    for(; step_id <= seqLen/2; step_id *= 2){
        substep_id = step_id;
        
        for(; substep_id >= 1; substep_id /= 2){
    
            if((t_id / substep_id) % 2 == 0){

                if((t_id / (2*step_id)) % 2 == 0){
                    if(edge_list[t_id].target_point > edge_list[t_id+substep_id].target_point||
                        (edge_list[t_id].target_point == edge_list[t_id+substep_id].target_point&&
                            edge_list[t_id].distance > edge_list[t_id+substep_id].distance)){

                        temparory_edge = edge_list[t_id];
                        edge_list[t_id] = edge_list[t_id+substep_id];
                        edge_list[t_id+substep_id] = temparory_edge;

                    }
                }else{
                    if(edge_list[t_id].target_point < edge_list[t_id+substep_id].target_point||
                        (edge_list[t_id].target_point == edge_list[t_id+substep_id].target_point&&
                            edge_list[t_id].distance < edge_list[t_id+substep_id].distance)){

                        temparory_edge = edge_list[t_id];
                        edge_list[t_id] = edge_list[t_id+substep_id];
                        edge_list[t_id+substep_id] = temparory_edge;

                    }
                }

            }

            __syncthreads();
                    
        }

        __syncthreads();

    }

    if(b_id*seqLen+t_id < num_of_edges){
        edgeGlobalDev[b_id*seqLen+t_id] = edge_list[t_id];
    }
}

extern "C" __global__ void mergeSubSeqonGpu(Edge* edgeGlobalDev, int num_of_edges, int seqLen){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;
    int seqIndex = glb_id/seqLen;
    int localIndex = glb_id%seqLen;

    Edge temparory_edge;
    int fstEdge = seqIndex*2*seqLen+localIndex;
    int secEdge = (seqIndex+1)*2*seqLen-localIndex-1;

    if(fstEdge < num_of_edges&&secEdge < num_of_edges){
        if(edgeGlobalDev[fstEdge].target_point > edgeGlobalDev[secEdge].target_point||
            (edgeGlobalDev[fstEdge].target_point == edgeGlobalDev[secEdge].target_point&&
                edgeGlobalDev[fstEdge].distance > edgeGlobalDev[secEdge].distance)){

            temparory_edge = edgeGlobalDev[fstEdge];
            edgeGlobalDev[fstEdge] = edgeGlobalDev[secEdge];
            edgeGlobalDev[secEdge] = temparory_edge;

        }
    }
}

extern "C" __global__ void initemparory_flagonGpu(Edge* edgeGlobalDev, int* flags, int num_of_edges, int virtualNum, int total_num_of_points){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;

    if(glb_id < num_of_edges){
        int curId = edgeGlobalDev[glb_id].target_point;
        int last_id = total_num_of_points;
        if(glb_id != 0){
            last_id = edgeGlobalDev[glb_id-1].target_point;
        }
        if(curId == last_id){
            flags[glb_id] = 0;
        }
        else{
            flags[glb_id] = 1;
        }
    
        if(glb_id == 0){
            flags[glb_id] = 1;
        }
    }else if(glb_id >= num_of_edges&&glb_id < virtualNum){
        flags[glb_id] = 0;
    }

}

extern "C" __global__ void sumInFlagonGpu(int* flags, int stride, int elementNum){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;

    if(glb_id < elementNum){
        int ai = stride*(2*glb_id+1)-1;
        int bi = stride*(2*glb_id+2)-1;  
        flags[bi] += flags[ai];
    }
}

extern "C" __global__ void sumInFlagDownonGpu(int* flags, int stride, int elementNum){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;

    if (glb_id < elementNum){
        int ai = stride*(2*glb_id+1)-1;    
        int bi = stride*(2*glb_id+2)-1;
        int t = flags[ai]; 
        flags[ai] = flags[bi];
        flags[bi] += t;       
    }

}

extern "C" __global__ void addUpFlagonGpu(int* a, int* b, int elementNum){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;

    if (glb_id < elementNum){
        a[glb_id] += b[glb_id];    
    }
}

extern "C" __global__ void generateInvertedList(int* flags, int* backupArray, int* indexArray, int num_of_edges, int validnum_of_edges){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int glb_id = b_id*blockDim.x+t_id;

    if(glb_id < num_of_edges){
        int curFlag = backupArray[glb_id];

        if(curFlag){
            int index = flags[glb_id];
            indexArray[index-1] = glb_id;
        }
    }

    if(glb_id == 0){
        indexArray[validnum_of_edges] = num_of_edges;
    }
}*/
