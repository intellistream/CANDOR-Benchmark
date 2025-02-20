//
// Created by LIUJUN on 19/1/2025.
//

#ifndef GANNSINDEX_H
#define GANNSINDEX_H
#include <CANDY/AbstractIndex.h>
namespace CANDY{
    extern int64_t DIM;
    extern int USE_L2_DIST_ ;
    extern int  USE_IP_DIST_;
    extern int  USE_COS_DIST_ ;
}
#include "../GANNSIndex/graph_index/navigable_small_world.h"
#include "../GANNSIndex/graph_index/hierarchical_navigable_small_world.h"
#include "../GANNSIndex/data.h"
namespace CANDY{

class GANNSIndex : public AbstractIndex{
public:
    int64_t vecDim = 0;
    GraphWrapper* graph;
    std::string graph_type = "hnsw";
    gData* points;
    std::vector<faiss::idx_t> Ids;
    GANNSIndex(std::string graph_type_);
    /**
    * @brief set the index-specific config related to one index
    * @param cfg the config of this class
    * @return bool whether the configuration is successful
    */
    virtual bool setConfig(INTELLI::ConfigMapPtr cfg);
 /**
* @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
* @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
* @param t the tensor, some index need to be single row
* @return bool whether the loading is successful
*/
 virtual bool loadInitialTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t);
 /**
  * @brief insert a tensor with Ids
  * @note This is majorly an online function
  * @param t the tensor, some index need to be single row
  * @return bool whether the insertion is successful
  */
 virtual bool insertTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t);
    /**
    * @brief search the k-NN of a query tensor, return their index
    * @param t the tensor, allow multiple rows
    * @param k the returned neighbors
    * @return std::vector<faiss::idx_t> the index, follow faiss's order
    */
    virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);

private:

};
}
#define newGANNSIndex std::make_shared<CANDY::GANNSIndex>
#endif //GANNSINDEX_H
