#if GANNS
if(CANDY::USE_L2_DIST_){

    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
}
else if (CANDY::USE_IP_DIST_){
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
}
else if(CANDY::USE_COS_DIST_){
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);

    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 16);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 8);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 4);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 2);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 1);

    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 16);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 8);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 4);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 2);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 1);
}

if(CANDY::USE_L2_DIST_){
    dist = sqrt(dist);
}
else if (CANDY::USE_IP_DIST_){
    dist = -dist;
}
else if(CANDY::USE_COS_DIST_){
    p_l2 = sqrt(p_l2);
    q_l2 = sqrt(q_l2);
    dist = dist / (p_l2 * q_l2);
    if (t_id == 0) {
        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }
    }
}
#endif