#pragma once

#include <memory>
#include <vector>
#include <string.h>
#include <fstream>

using namespace std;

class gData{

private:
    float* data_;
    int num_of_points_;
    int dim_of_point_;

    void ReadVectorsFromFiles(string path){
        ifstream in_descriptor(path, std::ios::binary);
        
        if (!in_descriptor.is_open()) {
            exit(1);
        }

        in_descriptor.read((char*)&dim_of_point_, 4);

        in_descriptor.seekg(0, std::ios::end);
        long long file_size = in_descriptor.tellg(); 
        num_of_points_ = file_size / (dim_of_point_ + 1) / 4;

        data_ = new float[dim_of_point_ * num_of_points_];
        //memset(data_, 0, 4 * num_of_points_ * dim_of_point_);
    
        in_descriptor.seekg(0, std::ios::beg);

        for (int i = 0; i < num_of_points_; i++) {
            in_descriptor.seekg(4, std::ios::cur);
            in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_ * 4);
        }

        in_descriptor.close();
    }

public:
    gData(string path){
        ReadVectorsFromFiles(path);
    }
    gData(torch::Tensor &t){
        num_of_points_ = t.size(0);
        dim_of_point_ = t.size(1);
        data_ = new float[num_of_points_ * dim_of_point_];
        memcpy(data_, t.contiguous().data_ptr<float>(), num_of_points_ * dim_of_point_ * 4);
    }
    int appendPoints(torch::Tensor &t)
    {
        const int res = num_of_points_;
        // 计算追加后的新点数
        const int new_num_of_points = num_of_points_ + t.size(0);

        // 扩展内存: 使用新的内存块来保存现有数据和新数据
        float* new_data = new float[new_num_of_points * dim_of_point_];

        // 复制原有数据到新的内存块
        memcpy(new_data, data_, num_of_points_ * dim_of_point_ * sizeof(float));

        // 复制新数据到新的内存块
        memcpy(new_data + num_of_points_ * dim_of_point_, t.contiguous().data_ptr<float>(), t.size(0) * dim_of_point_ * sizeof(float));

        // 释放原内存
        delete[] data_;

        // 更新 data_ 指向新内存
        data_ = new_data;

        // 更新点的数量
        num_of_points_ = new_num_of_points;

        return res;
    }
    float* GetFirstPositionofPoint(int point_id) const{
        return data_ + point_id * dim_of_point_;
    }
    float compare_(float* a, float* b) const {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            float diff = a[i] - b[i];
            dist += diff ;
        }
        return dist;
    }
    float L2Distance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return sqrt(dist);
    }
    
    float IPDistance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            dist -= a[i] * b[i];
        }
        return dist;
    }
    
    float COSDistance(float* a, float* b) {
        float dist = 0;
        float length_a = 0, length_b = 0;

        for (int i = 0; i < dim_of_point_; ++i) {
            dist += a[i] * b[i];
            length_a += a[i] * a[i];
            length_b += b[i] * b[i];
        }

        dist = dist / sqrt(length_a) / sqrt(length_b);

        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }

        return dist;
    }

    inline int GetNumPoints(){
        return num_of_points_;
    }

    int GetDimofPoints(){
        return dim_of_point_;
    }

};