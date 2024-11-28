//
// Created by rubato on 12/6/24.
//
#include<CANDY/DAGNNIndex/DAGNN.h>
#include <faiss/utils/Heap.h>


#define chronoElapsedTime(start)                               \
    std::chrono::duration_cast<std::chrono::microseconds>(     \
            std::chrono::high_resolution_clock::now() - start) \
            .count()

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */


/// compute var after concerning Xn+1
double add_zero_to_var(size_t n, double variance, double sum) {
    double old_avg = 0.0;
    if(n>1) {
        old_avg = sum/((n-1)*1.0);
    }
    double new_avg = sum/(n*1.0);
    double new_variance = (n-1)*1.0/(n*1.0)*(variance+old_avg*old_avg/n);
    return new_variance;

}
/// compute var after replacing Xi from old_value to new_value
double update_var_with_new_value(size_t n, double variance, double sum, double new_value, double old_value) {
    auto old_avg = sum*1.0/(n*1.0);
    auto new_avg = old_avg+(new_value-old_value)*1.0/(n*1.0);
    auto new_variance = variance + (new_avg-old_avg)*(new_avg-old_avg)+((new_value-new_avg)*(new_value-new_avg)-(old_value-new_avg)*(old_value-new_avg))/n*1.0;

    return new_variance;
}

double combine_var(double old_sum, double new_sum, double old_var, double new_var, size_t prev_ntotal, size_t new_ntotal) {
    double avg_prev=0.0;
    if(prev_ntotal!=0) {
        avg_prev = old_sum/(prev_ntotal*1.0);
    }
    double avg_new = new_sum/(new_ntotal*1.0);
    double term1 = (prev_ntotal-1.0)*old_var;
    double term2 = (new_ntotal-1.0)*new_var;
    double term3 =  (avg_new-avg_prev)*(avg_new-avg_prev);
    double bigterm1 = (term1+term2)/(prev_ntotal+new_ntotal-1.0);
    double bigterm2 = prev_ntotal*new_ntotal*1.0*term3/((prev_ntotal+new_ntotal)*(prev_ntotal+new_ntotal-1)*1.0);
    double variance = bigterm1+bigterm2;

    return variance;

}

void CANDY::DynamicTuneHNSW::changeEdge(CANDY::DynamicTuneHNSW::idx_t src, CANDY::DynamicTuneHNSW::idx_t old_dest,
                                        CANDY::DynamicTuneHNSW::idx_t new_dest, float distance) {
    auto node_src = linkLists[src];
    auto neighbor_src = node_src->neighbors[0];
    auto distance_src = node_src->distances[0];
    double distance_sum = 0.0;
    double previous_distance;
    for(int i=0; i< nb_neighbors(0); i++){
        if(neighbor_src[i]==-1){
            break;
        }
        distance_sum += distance_src[i];
        if(neighbor_src[i]==old_dest){
            previous_distance = distance_src[i];
            neighbor_src[i] = new_dest;
            distance_src[i]=distance;
        }
    }
    /// no degree change, only update neighbor distance metric;
    auto previous_distance_avg = distance_sum/(node_src->bottom_connections*1.0);
    auto current_distance_avg = (distance_sum-previous_distance+distance)/(node_src->bottom_connections*1.0);
    float old_variance = graphStates.global_stat.neighbor_distance_variance;
    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
    auto n = graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
            n, graphStates.global_stat.neighbor_distance_variance,
            graphStates.global_stat.neighbor_distance_sum,
            current_distance_avg, previous_distance_avg);
    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);

}

void CANDY::DynamicTuneHNSW::removeDuplicateEdge(CANDY::DynamicTuneHNSW::idx_t src) {
    auto node_src = linkLists[src];
    auto neighbor_src = node_src->neighbors[0];
    auto distance_src = node_src->distances[0];
    double distance_sum = 0.0;
    std::unordered_set<idx_t> has;
    for(int i=0; i<nb_neighbors(0); i++){
        if(neighbor_src[i]==-1){
            break;
        }
        distance_sum += distance_src[i];
        if(!has.contains(neighbor_src[i])){
            has.insert(neighbor_src[i]);
        } else {
            neighbor_src[i] = -2;
        }
    }
    auto prev_degree = node_src->bottom_connections;
    auto previous_distance_avg = distance_sum/(node_src->bottom_connections*1.0);
    distance_sum = 0.0;
    int j=0;
    for(int i=0; i<nb_neighbors(0); i++){
        if(neighbor_src[i]==-1){
            break;
        }
        if(neighbor_src[i]!=-2){
            distance_sum += distance_src[i];
            neighbor_src[j] = neighbor_src[i];
            distance_src[j++] = distance_src[i];
        }
    }
    node_src->bottom_connections = j;
    auto current_degree = node_src->bottom_connections;
    auto current_distance_avg = (distance_sum)/(node_src->bottom_connections*1.0);
    float old_variance = graphStates.global_stat.neighbor_distance_variance;
    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
    auto n = graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
    graphStates.global_stat.degree_variance = update_var_with_new_value(
            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
            current_degree, prev_degree);
    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);


    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
            n, graphStates.global_stat.neighbor_distance_variance,
            graphStates.global_stat.neighbor_distance_sum,
            current_distance_avg, previous_distance_avg);
    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
}

void CANDY::DynamicTuneHNSW::randomPickAction(){
    std::mt19937 rng(std::time(nullptr));
    //std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::uniform_int_distribution<int> firstRange(0, 8);
    //std::uniform_int_distribution<int> secondRange(11, 46);


    //double p = probDist(rng);
    int selectedNumber;
    if (1) {

        selectedNumber = firstRange(rng);
    } else {

        //selectedNumber = secondRange(rng);
    }

    printf("performing %ld\n", selectedNumber);
    performAction(selectedNumber);
}

void CANDY::DynamicTuneHNSW::updateGlobalState() {
    /// Updating global states from local states
    auto prev_ntotal = graphStates.global_stat.ntotal;
    auto new_ntotal = graphStates.time_local_stat.ntotal;
    //printf("hierarchy vertices evictions:%ld\n", graphStates.window_states.hierarchyVertices.evictions);
    /// Value
    {
        for(int64_t d=0; d<vecDim; d++) {
            auto prev = graphStates.global_stat.value_average[d];
            auto newv = graphStates.time_local_stat.value_average[d];
            graphStates.global_stat.value_variance[d]=combine_var(prev_ntotal*prev, new_ntotal*newv, graphStates.global_stat.value_variance[d], graphStates.time_local_stat.value_variance[d], prev_ntotal, new_ntotal);
            graphStates.global_stat.value_average[d] = (prev_ntotal*prev + new_ntotal*newv)/(prev_ntotal + new_ntotal);

        }
    }
    /// Connectivity
    {
        graphStates.global_stat.degree_sum = graphStates.time_local_stat.degree_sum_old + graphStates.time_local_stat.degree_sum_new;
        graphStates.global_stat.degree_variance = combine_var(
             graphStates.time_local_stat.degree_sum_old, graphStates.time_local_stat.degree_sum_new,
             graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_variance_new,
             prev_ntotal, new_ntotal);
        /// Combine variance

        graphStates.global_stat.neighbor_distance_sum = graphStates.time_local_stat.neighbor_distance_sum_old+graphStates.time_local_stat.neighbor_distance_sum_new;
        graphStates.global_stat.neighbor_distance_variance = combine_var(
            graphStates.time_local_stat.neighbor_distance_sum_old, graphStates.time_local_stat.neighbor_distance_sum_new,
            graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_variance_new,
            prev_ntotal, new_ntotal);
        if(verbose) {
            printf("UPdated previous degree var %lf avg %lf, new data degree var %lf avg %lf\n", graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_sum_old/(prev_ntotal*1.0), graphStates.time_local_stat.degree_variance_new, graphStates.time_local_stat.degree_sum_new/(new_ntotal*1.0));
            printf("UPdated previous neighbor distance var %lf , new data neighbor distance var %lf\n", graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_variance_new);
        }

    }

    /// Navigability
    {
        graphStates.global_stat.steps_taken_avg = (graphStates.global_stat.steps_taken_avg*prev_ntotal + graphStates.time_local_stat.steps_taken_sum)/((prev_ntotal+new_ntotal)*1.0);
        if(verbose) {
            printf("THis batch steps taken %ld and max %ld and avg %f, global max %ld global avg %lf\n", graphStates.time_local_stat.steps_taken_sum, graphStates.time_local_stat.steps_taken_max, graphStates.time_local_stat.steps_taken_sum/(graphStates.time_local_stat.ntotal*1.0), graphStates.global_stat.steps_taken_max, graphStates.global_stat.steps_taken_avg);
        }
    }
    //printf("ALl old vertices in window:\n");
    //graphStates.window_states.oldVertices.print_all();
    ///Window
    {
        auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
        auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
        if(graphStates.global_stat.ntotal >1000) {
            for (idx_t i = prev_ntotal; i < prev_ntotal + new_ntotal; i++) {
                auto nb_connections = linkLists[i]->bottom_connections;
                if (nb_connections * 1.0 > degree_avg + dynamicParams.degree_std_range * std_dev + 1 ||
                    nb_connections * 1.0 < degree_avg - dynamicParams.degree_std_range * std_dev - 1) {
                    omp_set_lock(&state_lock);
                    graphStates.window_states.newVertices.insert(i, linkLists[i]);
                    omp_unset_lock(&state_lock);
                }
            }
        }
    }

    /// randomly pick a state to perform

    if(is_datamining && graphStates.time_local_stat.old_ntotal > 1000){
        randomPickAction();
    }

    //graphStates.window_states.reset();

    bd_stats.reset();
}

void CANDY::DynamicTuneHNSW::add(idx_t n, float* x) {
    // here params are already set or updated
    graphStates.window_states.reset();
    auto insertion_start = std::chrono::high_resolution_clock::now();
    assert(n>0);
    assert(x);

    assign_levels(n);
    idx_t n0 = storage->ntotal;
    storage->add(n,x);
    if(n0>0){

        graphStates.forward();

    }
    std::vector<idx_t> order(n);
    std::vector<idx_t> hist;
    { // make buckets with vectors of the same level
        // build histogram
        for (int i = 0; i < n; i++) {
            idx_t pt_id = i + n0;
            int pt_level = linkLists[pt_id]->level;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            idx_t pt_id = i + n0;
            int pt_level = linkLists[pt_id]->level;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    for(idx_t i=0; i<n; i++) {
        for(int64_t d=0; d<vecDim; d++) {
                auto new_value = x[vecDim*i+d];
                auto old_mean = graphStates.time_local_stat.value_average[d];
                auto m = (new_value-old_mean)/((i+1)*1.0);

                auto old_variance = graphStates.time_local_stat.value_variance[d];

                graphStates.time_local_stat.value_average[d] += m;
                auto new_variance =((i)*1.0/((i+1)*1.0))*(old_variance+((old_mean-new_value)*(old_mean-new_value)/((i+1)*1.0)));
                graphStates.time_local_stat.value_variance[d] =new_variance;

        }
    }
    std::vector<omp_lock_t> locks(n0+n);
    for(idx_t i=0; i<n0+n; i++){
        omp_init_lock(&locks[i]);
    }
    omp_init_lock(&state_lock);
    /// adding
    {
        idx_t i1 = n;
        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            idx_t i0 = i1 - hist[pt_level];
#pragma omp parallel if (i1>i0+50)
            {
                DAGNN::VisitedTable vt(n0 + n);
                std::unique_ptr<faiss::DistanceComputer> disq(
                        storage_distance_computer(storage));
#pragma omp for schedule(static)
                for (idx_t i = i0; i < i1; i++) {
                    // update measure
                    omp_set_lock(&state_lock);
                    graphStates.time_local_stat.ntotal += 1;

                    graphStates.time_local_stat.degree_variance_new = add_zero_to_var(
                            graphStates.time_local_stat.ntotal, graphStates.time_local_stat.degree_variance_new,
                            graphStates.time_local_stat.degree_sum_new);
                    graphStates.time_local_stat.neighbor_distance_variance_new = add_zero_to_var(
                            graphStates.time_local_stat.ntotal,
                            graphStates.time_local_stat.neighbor_distance_variance_new,
                            graphStates.time_local_stat.neighbor_distance_sum_new);
                    timestamp++;
                    omp_unset_lock(&state_lock);

                    idx_t id = order[i];
                    disq->set_query(x + (id - n0) * vecDim);
                    auto node = linkLists[id ];
                    last_visited[i + n0] = timestamp;
                    greedy_insert(*disq, *node, vt, locks);
                }
            }
            i1 = i0;
        }
        assert(i1 == 0);
    }
    for (idx_t i = 0; i < n0+n; i++) {
        omp_destroy_lock(&locks[i]);
    }
    graphStates.window_states.last_insertion_latency = chronoElapsedTime(insertion_start);



}



void CANDY::DynamicTuneHNSW::greedy_insert(faiss::DistanceComputer& disq, CANDY::DynamicTuneHNSW::Node& node, DAGNN::VisitedTable& vt, std::vector<omp_lock_t>& locks){
    /// Greedy Phase
    idx_t nearest = -1;
    auto assigned_level = node.level;
    if(entry_points.empty()) {
        entry_points.push_back(node.id);
        max_level = assigned_level;
    }


    //printf("node %ld assigned level at %ld\n", node.id, node.level);
    omp_set_lock(&locks[node.id]);
    if(assigned_level > max_level) {
        node.level+=1;
        assigned_level +=1;
        max_level = assigned_level;
        auto l_neighbors = std::vector<idx_t>(nb_neighbors(max_level),-1);
        auto l_distances = std::vector<float>(nb_neighbors(max_level), -0.0);
        node.neighbors.push_back(l_neighbors);
        node.distances.push_back(l_distances);
        entry_points[0] = node.id;
        printf("changing max level to %ld\n", max_level);
    }
    nearest = entry_points[0];
    float dist_nearest = disq(nearest);
    if(assigned_level>0) {
        //printf("%ld with level=%ld\n", node.id, assigned_level);
    }

    int64_t steps_taken = 0;
    int64_t steps_expansion;
    for(size_t l=max_level; l>assigned_level; l--) {
        greedy_insert_top(disq, l, nearest, dist_nearest, steps_taken);
    }


    for(size_t l=assigned_level; l>0; l--) {
        std::priority_queue<CandidateCloser> candidates;
        greedy_insert_upper(disq, l, nearest, dist_nearest, candidates, steps_taken);
        /// Candidate Phase ON TOP LEVELS and linking
        auto entry_this_level = CandidateCloser(dist_nearest, nearest);
        //printf("pushing %ld with %f to candidates on level %ld\n", nearest, dist_nearest, l);
        candidates.push(entry_this_level);
        link_from_lock(disq, node.id, l, nearest, dist_nearest, candidates, vt, locks.data());


    }
    std::priority_queue<CandidateCloser> candidates;
    greedy_insert_base(disq, nearest, dist_nearest, candidates, steps_taken);
    // Navigability states
    graphStates.time_local_stat.steps_taken_sum += steps_taken;
    if(steps_taken > graphStates.time_local_stat.steps_taken_max) {
        if(steps_taken - dynamicParams.steps_above_avg > graphStates.global_stat.steps_taken_avg
            || steps_taken - dynamicParams.steps_above_max > graphStates.global_stat.steps_taken_max) {
            auto far_node = linkLists[nearest];
            omp_set_lock(&state_lock);
            graphStates.window_states.oldVertices.insert(nearest, far_node);
            omp_unset_lock(&state_lock);
        }
        graphStates.time_local_stat.steps_taken_max = steps_taken;


    }
    graphStates.global_stat.steps_taken_max = (graphStates.global_stat.steps_taken_max>steps_taken)?graphStates.global_stat.steps_taken_max:steps_taken;
    auto entry_base_level = CandidateCloser(dist_nearest, nearest);
    candidates.push(entry_base_level);
    link_from_base_lock(disq, node.id,  nearest, dist_nearest, candidates, vt, locks.data());
    /// Candidate phase on base level and linking

    omp_unset_lock(&locks[node.id]);






}



void CANDY::DynamicTuneHNSW::greedy_insert_top(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, int64_t& steps_taken){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        last_visited[nearest] = timestamp;
        size_t nb_neighbor_level = nb_neighbors(level);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            if(level > node->level) {
                break;
            }
            auto visiting = node->neighbors[level][i];

            if(visiting < 0) {
                break;
            }
            omp_set_lock(&state_lock);
            if(level == 1){
                if(timestamp - last_visited[visiting]>dynamicParams.expiration_timestamp){
                    if(linkLists[visiting]->level==1){

                        graphStates.window_states.hierarchyVertices.insert(visiting, linkLists[visiting]);

                    }
                }
            }
            omp_unset_lock(&state_lock);


            auto dist = disq(visiting);
            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                steps_taken += std::pow(32, level);
                bd_stats.steps_greedy++;
              //  printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_insert_upper(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        last_visited[nearest] = timestamp;
        size_t nb_neighbor_level = nb_neighbors(level);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            if(level > node->level) {
                break;
            }
            auto visiting = node->neighbors[level][i];
            if(visiting < 0) {
                break;
            }
            omp_set_lock(&state_lock);
            if(level == 1){
                if(timestamp - last_visited[visiting]>dynamicParams.expiration_timestamp){
                    if(linkLists[visiting]->level==1){

                        graphStates.window_states.hierarchyVertices.insert(visiting, linkLists[visiting]);

                    }
                }
            }
            omp_unset_lock(&state_lock);


            auto dist = disq(visiting);
            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                steps_taken += std::pow(32, level);
                bd_stats.steps_greedy++;
                //printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_insert_base(faiss::DistanceComputer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken){
    auto degree_avg = graphStates.global_stat.ntotal==0?0:graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
    auto std_dev = std::sqrt(graphStates.global_stat.degree_variance);
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        last_visited[nearest] = timestamp;
        if(node->bottom_connections*1.0>degree_avg+dynamicParams.degree_std_range*std_dev+1 || node->bottom_connections*1.0<degree_avg-dynamicParams.degree_std_range*std_dev-1) {
            omp_set_lock(&state_lock);
            graphStates.window_states.oldVertices.insert(node->id, node);
            omp_unset_lock(&state_lock);

        }
        size_t nb_neighbor_level = nb_neighbors(0);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            auto visiting = node->neighbors[0][i];
            if(visiting < 0) {
                break;
            }
            omp_set_lock(&state_lock);
            if(linkLists[visiting]->level==0) {
                if(linkLists[visiting]->bottom_connections*1.0>degree_avg+dynamicParams.degree_lift_range*std_dev+1) {
                    //graphStates.window_states.oldVertices.remove(visiting);

                    graphStates.window_states.hierarchyVertices.insert(visiting, linkLists[visiting]);

                }
            }
            omp_unset_lock(&state_lock);
            auto dist = disq(visiting);

            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                steps_taken += 1;
                bd_stats.steps_greedy_base++;
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}
/// We need the nearest point to consult while linking
void CANDY::DynamicTuneHNSW::link_from(faiss::DistanceComputer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt){
    /// Candidate phase
    priority_queue<CandidateFarther> selection;
    candidate_select(disq, level, selection, candidates, vt);
    /// Link Phase
    /// Prune this neighbor's candidate
    prune(disq, level, candidates);
    std::vector<idx_t> neighbors;
    neighbors.reserve(candidates.size());

    while(!candidates.empty()) {
        idx_t other_id = candidates.top().id;
        add_link(disq, idx, other_id, level);
        neighbors.push_back(other_id);
        candidates.pop();


    }

    for(auto nei: neighbors) {
        add_link(disq, nei, idx, level);

    }

}

void CANDY::DynamicTuneHNSW::link_from_lock(faiss::DistanceComputer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt, omp_lock_t* locks){
    /// Candidate phase
    priority_queue<CandidateFarther> selection;
    candidate_select(disq, level, selection, candidates, vt);
    /// Link Phase
    /// Prune this neighbor's candidate
    prune(disq, level, candidates);
    std::vector<idx_t> neighbors;
    neighbors.reserve(candidates.size());

    while(!candidates.empty()) {
        idx_t other_id = candidates.top().id;
        add_link(disq, idx, other_id, level);
        neighbors.push_back(other_id);
        candidates.pop();


    }
    omp_unset_lock(&locks[idx]);

    for(auto nei: neighbors) {
        omp_set_lock(&locks[nei]);
        add_link(disq, nei, idx, level);
        omp_unset_lock(&locks[nei]);

    }
    omp_set_lock(&locks[idx]);

}

/// We need the nearest point to consult while linking
void CANDY::DynamicTuneHNSW::link_from_base(faiss::DistanceComputer& disq, idx_t idx, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt){
    /// Candidate phase
    priority_queue<CandidateFarther> selection;
    candidate_select_base(disq, selection, candidates, vt);
    /// Link Phase
    /// Prune this neighbor's candidate
    prune_base(disq, candidates);
    std::vector<idx_t> neighbors;
    neighbors.reserve(candidates.size());
    int64_t current_degree;
    int64_t previous_degree;
    while(!candidates.empty()) {
        idx_t other_id = candidates.top().id;
        add_link_base(disq, idx, other_id, previous_degree, current_degree);
        neighbors.push_back(other_id);
        candidates.pop();

    }
    double new_degree_avg = graphStates.time_local_stat.degree_sum_new/(graphStates.time_local_stat.ntotal*1.0);
    double new_std_dev = std::sqrt(graphStates.time_local_stat.degree_variance_new);
    // /// TODO: Pre pruning to avoid further pruning
     if(current_degree<new_degree_avg-new_std_dev-1 || current_degree > new_degree_avg+new_std_dev+1) {
         omp_set_lock(&state_lock);
         graphStates.window_states.newVertices.insert(idx, linkLists[idx]);
         omp_unset_lock(&state_lock);
     }
    for(auto nei: neighbors) {
        add_link_base(disq, nei, idx, previous_degree, current_degree);
    }

}

void CANDY::DynamicTuneHNSW::link_from_base_lock(faiss::DistanceComputer& disq, idx_t idx, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt, omp_lock_t* locks){
    /// Candidate phase
    priority_queue<CandidateFarther> selection;
    candidate_select_base(disq, selection, candidates, vt);
    /// Link Phase
    /// Prune this neighbor's candidate
    prune_base(disq, candidates);
    std::vector<idx_t> neighbors;
    neighbors.reserve(candidates.size());
    int64_t current_degree;
    int64_t previous_degree;
    while(!candidates.empty()) {
        idx_t other_id = candidates.top().id;
        add_link_base(disq, idx, other_id, previous_degree, current_degree);
        neighbors.push_back(other_id);
        candidates.pop();

    }
    double new_degree_avg = graphStates.time_local_stat.degree_sum_new/(graphStates.time_local_stat.ntotal*1.0);
    double new_std_dev = std::sqrt(graphStates.time_local_stat.degree_variance_new);
    // /// TODO: Pre pruning to avoid further pruning
    if(current_degree<new_degree_avg-new_std_dev-1 || current_degree > new_degree_avg+new_std_dev+1) {
        omp_set_lock(&state_lock);
        graphStates.window_states.newVertices.insert(idx, linkLists[idx]);
        omp_unset_lock(&state_lock);
    }
    omp_unset_lock(&locks[idx]);
    for(auto nei: neighbors) {
        omp_set_lock(&locks[nei]);
        add_link_base(disq, nei, idx, previous_degree, current_degree);
        omp_unset_lock(&locks[nei]);
    }
    omp_set_lock(&locks[idx]);

}


void CANDY::DynamicTuneHNSW::candidate_select(faiss::DistanceComputer& disq, size_t level,  std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates,  DAGNN::VisitedTable& vt){
    /// candidates might contain long-edge from greedy phase
    if(!candidates.empty()){
        selection.emplace(candidates.top().dist, candidates.top().id);
    }
    vt.set(candidates.top().id);
    while(!selection.empty()){
        auto curr = selection.top();
        if(curr.dist > candidates.top().dist){
            break;
        }
        int curr_id = curr.id;
        selection.pop();
        auto curr_node = linkLists[curr_id];
        size_t neighbor_length = nb_neighbors(level);
        auto curr_linkList = curr_node->neighbors[level];
        //auto curr_distList = curr_node->distances[level];
        for(size_t i=0; i<neighbor_length; i++){
            auto expansion = curr_linkList[i];
            if(expansion<0){
                break;
            }
            if(vt.get(expansion)){
                continue;
            }
            if(deleteLists[expansion]){
                continue;
            }
            vt.set(expansion);

            float dis = disq(expansion);
            bd_stats.steps_expansion++;


            //float dis=curr_distList[i];


            if(candidates.size() < dynamicParams.efConstruction || candidates.top().dist > dis){
                candidates.emplace(dis, expansion);
                selection.emplace(dis, expansion);
                if(candidates.size()>dynamicParams.efConstruction){
                    candidates.pop();
                }
            }
        }
    }
    vt.advance();

}

void CANDY::DynamicTuneHNSW::candidate_select_base(faiss::DistanceComputer& disq, std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates,  DAGNN::VisitedTable& vt){
    /// candidates might contain long-edge from greedy phase
    if(!candidates.empty()){
        selection.emplace(candidates.top().dist, candidates.top().id);
    }
    int level = 0;
    vt.set(candidates.top().id);
    int64_t optimistic_count = 0;
    while(!selection.empty()){
        auto curr = selection.top();
        if(curr.dist > candidates.top().dist){
            break;
        }
        int curr_id = curr.id;
        selection.pop();
        auto curr_node = linkLists[curr_id];
        size_t neighbor_length = nb_neighbors(level);
        auto curr_linkList = curr_node->neighbors[level];
        auto curr_distList = curr_node->distances[level];
        for(size_t i=0; i<neighbor_length; i++){
            auto expansion = curr_linkList[i];
            if(expansion<0){
                break;
            }
            if(vt.get(expansion)){
                continue;
            }

            if(deleteLists[expansion]){
                continue;
            }
            vt.set(expansion);
            float dis;

            if(optimistic_count<dynamicParams.optimisticN) {
                dis= curr_distList[i];
                optimistic_count++;
            } else {
                dis = disq(expansion);
                bd_stats.steps_expansion_base++;
            }

            //float dis=curr_distList[i];

            if(candidates.size() < dynamicParams.efConstruction || candidates.top().dist > dis){
                candidates.emplace(dis, expansion);
                selection.emplace(dis, expansion);
                if(candidates.size()>dynamicParams.efConstruction){
                    candidates.pop();
                }
            }
        }
    }
    vt.advance();

}


void CANDY::DynamicTuneHNSW::prune(faiss::DistanceComputer& disq,size_t level,  std::priority_queue<CandidateCloser>& candidates){
    /// default version, selectively transfer candidates to output and then copy output to candidates
    int M = nb_neighbors(level);

    // if(level == 0) {
    //     M = (dynamicParams.bottom_connections_lower_bound+dynamicParams.bottom_connections_upper_bound)/2;
    // }
    //TODO: modify
    if(candidates.size()<M) {
        return;
    }
    bd_stats.prune_times++;
    std::vector<CandidateCloser> output;
    int count = 0;
    std::priority_queue<CandidateCloser> candidates_reverse;
    while(!candidates.empty()) {
        candidates_reverse.emplace(-candidates.top().dist, candidates.top().id);
        candidates.pop();
    }
    // TODO: MASSIVE DCO ON UPPER LEVELS
    int i=0;
    int j=0;
    while(!candidates_reverse.empty()){
        auto v1 = candidates_reverse.top();
        candidates_reverse.pop();
        j++;
        float dist_v1_q = -v1.dist;

        bool rng_good=true;
        for(auto v2: output){

            float dist_v1_v2 = disq.symmetric_dis(v1.id, v2.id);
            bd_stats.steps_pruning++;
            i++;
            //printf("v1=%f v2=%f v1_v2=%f\n", v1.dist,v2.dist,dist_v1_v2);
            // from DiskANN Vanama
            if(dist_v1_v2*dynamicParams.rng_alpha < dist_v1_q ){
                rng_good = false;
                break;
            }
        }

        if(rng_good){
            output.emplace_back(v1);
            if(output.size()>=M){
                // transfer output to candidate
                for(auto k : output) {
                    count++;
                    candidates.emplace(-k.dist, k.id);
                }
                //printf("avg distance %f with size %d\n", dis_sum/count, count);
                //printf("rng_good candidate size %d, total dco %d, avg dco %lf\n", j,i,i/(j*1.0));
                return;
            }

        }
    }
    for(auto k : output) {
        count++;
        candidates.emplace(-k.dist, k.id);
    }
    //printf("/candidate size %d, total dco %d, avg dco %lf\n", j,i,i/(j*1.0));
    //printf("avg distance %f with size %d\n", dis_sum/count, count);
}

void CANDY::DynamicTuneHNSW::prune_base(faiss::DistanceComputer& disq, std::priority_queue<CandidateCloser>& candidates){
    /// default version, selectively transfer candidates to output and then copy output to candidates
    int level = 0;
    int M = nb_neighbors(level);

    // if(level == 0) {
    //     M = (dynamicParams.bottom_connections_lower_bound+dynamicParams.bottom_connections_upper_bound)/2;
    // }
    //TODO: modify
    if(candidates.size()<M) {
        return;
    }
    bd_stats.prune_times_base++;
    std::vector<CandidateCloser> output;
    int count = 0;
    std::priority_queue<CandidateCloser> candidates_reverse;
    while(!candidates.empty()) {
        candidates_reverse.emplace(-candidates.top().dist, candidates.top().id);
        candidates.pop();
    }
    while(!candidates_reverse.empty()){
        auto v1 = candidates_reverse.top();
        candidates_reverse.pop();
        float dist_v1_q = -v1.dist;

        bool rng_good=true;
        for(auto v2: output){

            float dist_v1_v2 = disq.symmetric_dis(v1.id, v2.id);
            bd_stats.steps_pruning_base++;
            //printf("v1=%f v2=%f v1_v2=%f\n", v1.dist,v2.dist,dist_v1_v2);

            // from DiskANN Vanama
            if(dist_v1_v2 < (dist_v1_q / dynamicParams.rng_alpha)){
                rng_good = false;
                break;
            }
        }

        if(rng_good){
            output.emplace_back(v1);
            if(output.size()>=M){
                // transfer output to candidate
                for(auto k : output) {
                    count++;
                    candidates.emplace(-k.dist, k.id);
                }
                //printf("avg distance %f with size %d\n", dis_sum/count, count);
                return;
            }

        }
    }
    for(auto k : output) {
        count++;
        candidates.emplace(-k.dist, k.id);
    }
    //printf("avg distance %f with size %d\n", dis_sum/count, count);
}


void breakpoint() {
    return;
}
void CANDY::DynamicTuneHNSW::add_link(faiss::DistanceComputer& disq, idx_t src, idx_t dest, size_t level) {
    if(src==dest) {
        return;
    }
    int nb_neighbors_level = nb_neighbors(level);
    auto src_linkList = &linkLists[src]->neighbors[level];
    auto src_dist = &linkLists[src]->distances[level];
    if((*src_linkList)[nb_neighbors_level-1]<0) {
        size_t i=nb_neighbors_level;
        while(i>0) {
            if((*src_linkList)[i-1]>=0) {
                break;
            }
            i--;
        }
        //printf("updating links for %ld: ", src);

        (*src_linkList)[i] = dest;
        /// TODO: weighted edges

        (*src_dist)[i] = disq.symmetric_dis(src, dest);



        int64_t prev_degree = i;
        int64_t current_degree = i+1;
        float previous_distance_sum = 0;

        for(int j=0; j<i; j++) {
            previous_distance_sum += (*src_dist)[j];
        }
        float current_distance_sum = previous_distance_sum + (*src_dist)[i];
        float current_distance_avg = current_distance_sum/(current_degree*1.0);
        float previous_distance_avg = 0.0;
        if(prev_degree>0) {
            previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
        }
        ///TODO: Update degree stats
        if(level==0 ) {
            linkLists[src]->bottom_connections = current_degree;
            if(src<graphStates.time_local_stat.old_ntotal) {
                omp_set_lock(&state_lock);
                auto n = graphStates.time_local_stat.old_ntotal;
                    graphStates.time_local_stat.degree_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_sum_old,
                        current_degree, prev_degree);
                    graphStates.time_local_stat.degree_sum_old += (current_degree-prev_degree);

                    graphStates.time_local_stat.neighbor_distance_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_sum_old,
                        current_distance_avg, previous_distance_avg);
                    graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);
                omp_unset_lock(&state_lock);


            } else {
                omp_set_lock(&state_lock);
                auto n = graphStates.time_local_stat.ntotal;
                if(n!=1) {
                        graphStates.time_local_stat.degree_variance_new = update_var_with_new_value(
                            n, graphStates.time_local_stat.degree_variance_new, graphStates.time_local_stat.degree_sum_new,
                            current_degree, prev_degree);
                        graphStates.time_local_stat.neighbor_distance_variance_new = update_var_with_new_value(
                            n, graphStates.time_local_stat.neighbor_distance_variance_new, graphStates.time_local_stat.neighbor_distance_sum_new,
                            current_distance_avg, previous_distance_avg);

                }
                graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
                graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);
                omp_unset_lock(&state_lock);
            }
        }
        return;
    }
    std::priority_queue<CandidateCloser> final_neighbors;


    final_neighbors.emplace(disq.symmetric_dis(src, dest), dest);

    double current_distance_sum = 0.0;
    double previous_distance_sum = 0.0;
    for(size_t i=0; i<nb_neighbors_level; i++) {
        auto neighbor  = (*src_linkList)[i];
        /// TODO: WEIGHTED EDGES
        auto dist = (*src_dist)[i];
        final_neighbors.emplace(dist, neighbor);
        previous_distance_sum += dist;

    }

    prune(disq, level, final_neighbors);

    size_t i=0;


    while(!final_neighbors.empty()) {
        //printf("%ld ", final_neighbors.top().id);
        (*src_linkList)[i] = final_neighbors.top().id;
        (*src_dist)[i++] = final_neighbors.top().dist;
        current_distance_sum+=final_neighbors.top().dist;
        final_neighbors.pop();
    }
    //printf("after pruning %ld avg distance = %f with size %d\n\n", src, dis_sum/count, count);
    //printf("\n");
    int64_t current_degree = i;

    int64_t prev_degree = nb_neighbors(level);
    float current_distance_avg = current_distance_sum/(current_degree*1.0);
    float previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
    if(level==0 ) {
        linkLists[src]->bottom_connections = current_degree;
        if(src<graphStates.time_local_stat.old_ntotal) {
            omp_set_lock(&state_lock);
            auto n = graphStates.time_local_stat.old_ntotal;
                graphStates.time_local_stat.degree_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_sum_old,
                        current_degree, prev_degree);
                graphStates.time_local_stat.degree_sum_old+=(current_degree-prev_degree);


                graphStates.time_local_stat.neighbor_distance_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_sum_old,
                        current_distance_avg, previous_distance_avg);
            graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);
            omp_unset_lock(&state_lock);
        } else {
            omp_set_lock(&state_lock);
            auto n = graphStates.time_local_stat.ntotal;
            if(n!=1) {
                graphStates.time_local_stat.degree_variance_new = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_new, graphStates.time_local_stat.degree_sum_new,
                        current_degree, prev_degree);
                graphStates.time_local_stat.neighbor_distance_variance_new = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_new, graphStates.time_local_stat.neighbor_distance_sum_new,
                        current_distance_avg, previous_distance_avg);
            }

            graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
            graphStates.time_local_stat.neighbor_distance_sum_new+=(current_distance_avg-previous_distance_avg);
            omp_unset_lock(&state_lock);
        }

    }
    while(i<nb_neighbors_level) {
        (*src_linkList)[i++]=-1;
    }
}

void CANDY::DynamicTuneHNSW::add_link_base(faiss::DistanceComputer& disq, idx_t src, idx_t dest, int64_t& prev_degree, int64_t& current_degree) {
    if(src==dest) {
        return;
    }
    int level = 0;
    int nb_neighbors_level = nb_neighbors(level);
    auto src_linkList = &linkLists[src]->neighbors[level];
    auto src_dist = &linkLists[src]->distances[level];
    if((*src_linkList)[nb_neighbors_level-1]<0) {
        size_t i=nb_neighbors_level;
        while(i>0) {
            if((*src_linkList)[i-1]!=-1) {
                break;
            }
            i--;
        }
        //printf("updating links for %ld: ", src);

        (*src_linkList)[i] = dest;


        (*src_dist)[i] = disq.symmetric_dis(src, dest);



        prev_degree = i;
        current_degree = i+1;
        float previous_distance_sum = 0;

        for(int j=0; j<i; j++) {
            previous_distance_sum += (*src_dist)[j];
        }
        float current_distance_sum = previous_distance_sum + (*src_dist)[i];
        float current_distance_avg = current_distance_sum/(current_degree*1.0);
        float previous_distance_avg = 0.0;
        if(prev_degree>0) {
            previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
        }
        ///TODO: Update degree stats
        if(level==0 ) {
            linkLists[src]->bottom_connections = current_degree;
            if(src<graphStates.time_local_stat.old_ntotal) {
                omp_set_lock(&state_lock);
                auto n = graphStates.time_local_stat.old_ntotal;
                    graphStates.time_local_stat.degree_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_sum_old,
                        current_degree, prev_degree);
                    graphStates.time_local_stat.degree_sum_old += (current_degree-prev_degree);

                    graphStates.time_local_stat.neighbor_distance_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_sum_old,
                        current_distance_avg, previous_distance_avg);
                    graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);
                omp_unset_lock(&state_lock);
            } else {
                omp_set_lock(&state_lock);
                auto n = graphStates.time_local_stat.ntotal;
                if(n!=1) {
                        graphStates.time_local_stat.degree_variance_new = update_var_with_new_value(
                            n, graphStates.time_local_stat.degree_variance_new, graphStates.time_local_stat.degree_sum_new,
                            current_degree, prev_degree);
                        graphStates.time_local_stat.neighbor_distance_variance_new = update_var_with_new_value(
                            n, graphStates.time_local_stat.neighbor_distance_variance_new, graphStates.time_local_stat.neighbor_distance_sum_new,
                            current_distance_avg, previous_distance_avg);

                }
                graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
                graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);
                omp_unset_lock(&state_lock);
            }
        }
        return;
    }
    std::priority_queue<CandidateCloser> final_neighbors;


    final_neighbors.emplace(disq.symmetric_dis(src, dest), dest);

    double current_distance_sum = 0.0;
    double previous_distance_sum = 0.0;
    for(size_t i=0; i<nb_neighbors_level; i++) {
        auto neighbor  = (*src_linkList)[i];
        /// TODO: WEIGHTED EDGES
        auto dist = (*src_dist)[i];
        final_neighbors.emplace(dist, neighbor);
        previous_distance_sum += dist;

    }

    prune_base(disq,  final_neighbors);

    size_t i=0;


    while(!final_neighbors.empty()) {
        //printf("%ld ", final_neighbors.top().id);
        (*src_linkList)[i] = final_neighbors.top().id;
        (*src_dist)[i++] = final_neighbors.top().dist;
        current_distance_sum+=final_neighbors.top().dist;
        final_neighbors.pop();
    }
    //printf("after pruning %ld avg distance = %f with size %d\n\n", src, dis_sum/count, count);
    //printf("\n");
    current_degree = i;

    prev_degree = nb_neighbors(level);
    float current_distance_avg = current_distance_sum/(current_degree*1.0);
    float previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
    if(level==0 ) {
        linkLists[src]->bottom_connections = current_degree;
        if(src<graphStates.time_local_stat.old_ntotal) {
            omp_set_lock(&state_lock);
            auto n = graphStates.time_local_stat.old_ntotal;
                graphStates.time_local_stat.degree_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_sum_old,
                        current_degree, prev_degree);
                graphStates.time_local_stat.degree_sum_old+=(current_degree-prev_degree);


                graphStates.time_local_stat.neighbor_distance_variance_old = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_old, graphStates.time_local_stat.neighbor_distance_sum_old,
                        current_distance_avg, previous_distance_avg);
            graphStates.time_local_stat.neighbor_distance_sum_old += (current_distance_avg-previous_distance_avg);

            omp_unset_lock(&state_lock);


        } else {
            omp_set_lock(&state_lock);
            auto n = graphStates.time_local_stat.ntotal;
            if(n!=1) {
                graphStates.time_local_stat.degree_variance_new = update_var_with_new_value(
                        n, graphStates.time_local_stat.degree_variance_new, graphStates.time_local_stat.degree_sum_new,
                        current_degree, prev_degree);
                graphStates.time_local_stat.neighbor_distance_variance_new = update_var_with_new_value(
                        n, graphStates.time_local_stat.neighbor_distance_variance_new, graphStates.time_local_stat.neighbor_distance_sum_new,
                        current_distance_avg, previous_distance_avg);
            }

            graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
            graphStates.time_local_stat.neighbor_distance_sum_new+=(current_distance_avg-previous_distance_avg);
        }
        omp_unset_lock(&state_lock);
    }
    while(i<nb_neighbors_level) {
        (*src_linkList)[i++]=-1;
    }
}

void CANDY::DynamicTuneHNSW::search(faiss::DistanceComputer& disq, idx_t annk, idx_t* results, float* distances, DAGNN::VisitedTable& vt) {
    if(entry_points.empty()) {
        printf(" EMPTY INDEX!\n");
        return;
    }

    auto nearest = entry_points[0];

    float d_nearest = disq(nearest);


    for(size_t l=max_level; l>=1; l--) {
        std::priority_queue<CandidateCloser> candidates_greedy;
        greedy_search_upper(disq, l, nearest, d_nearest, candidates_greedy);

    }
    std::priority_queue<CandidateCloser> candidates_greedy;
    //greedy_search_base(disq,nearest,d_nearest,candidates_greedy);
    int64_t efSearch = annk>dynamicParams.efSearch? annk:dynamicParams.efSearch;
    DAGNN::MinimaxHeap candidates(efSearch);

    candidates.push(nearest, d_nearest);

    candidate_search(disq, 0, annk, results, distances,candidates,vt);
    vt.advance();

}

void CANDY::DynamicTuneHNSW::greedy_search_upper(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        size_t nb_neighbor_level = nb_neighbors(level);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            if(level > node->level) {
                break;
            }
            auto visiting = node->neighbors[level][i];
            if(visiting < 0) {
                break;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto dist = disq(visiting);
            auto temp = chronoElapsedTime(start);
            time_dco+= temp;
            time_dco_upper += temp;
            num_dco+=1;
            num_dco_upper+=1;

            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }

        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_search_base(faiss::DistanceComputer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        size_t nb_neighbor_level = nb_neighbors(0);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            auto visiting = node->neighbors[0][i];
            if(visiting < 0) {
                break;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto dist = disq(visiting);
            auto temp= chronoElapsedTime(start);
            time_dco+=temp;
            time_dco_base+=temp;
            num_dco+=1;
            num_dco_base+=1;

            //printf("trying to %ld with dist = %.2f on level 0 \n", visiting, dist);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level 0 \n", visiting, dist);
            }

        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

int CANDY::DynamicTuneHNSW::candidate_search(faiss::DistanceComputer& disq, size_t level, idx_t annk, idx_t* results, float* distances,DAGNN::MinimaxHeap& candidates, DAGNN::VisitedTable& vt) {
    idx_t nres = 0;

    int nstep = 0;
    for(int i=0; i<candidates.size(); i++) {
        idx_t v1=candidates.ids[i];
        float d = candidates.dis[i];
        if(nres<annk) {
            faiss::maxheap_push(++nres, distances, results, d, v1);
        } else if(d<distances[0]) {
            faiss::maxheap_replace_top(nres, distances, results, d,v1);
        }
        vt.set(v1);
    }

    while(candidates.size()>0) {
        float d0 = 0;
        idx_t v0 = candidates.pop_min(&d0);
        //printf("popping v0=%ld\n", v0);
        auto nb_neighbors_level = nb_neighbors(level);
        auto v0_linkList = linkLists[v0]->neighbors[level];
        auto v0_vector=get_vector(v0);

        //
//        for(size_t j=0; j<nb_neighbors_level; j++) {
//            idx_t v1 = v0_linkList[j];
//            //printf("stepping to %ld\n", v1);
//            if(v1<0) {
//                break;
//            }
//            auto vget = vt.get(v1);
//            if(vget) {
//                continue;
//            }
//            // do not consider deleted vertices
//            if(deleteLists[v1]){
//                continue;
//            }
//
//            vt.set(v1);
//
//            //saved_vectors[counter] = get_vector(v1);
//
//            //float d1 = v0_distList[j];
//            auto start = std::chrono::high_resolution_clock::now();
//            float d1=disq(v1);
//            auto temp = chronoElapsedTime(start);
//            time_dco+=temp;
//            time_dco_expansion+=temp;
//
//            num_dco+=1;
//            num_dco_expansion+=1;
//            //float d1_d0 = disq.distance(v1_vector, v0_vector);
//            //float d0=disq(v0_vector);
//            //printf("v1=%ld d0=%f d1=%f d1_d0=%f\n", v1, d0, d1, d1_d0);
//
//
//            if(nres<annk) {
//                faiss::maxheap_push(++nres, distances, results, d1, v1);
//            } else if(d1<distances[0]) {
//                faiss::maxheap_replace_top(nres, distances, results, d1, v1);
//            }
//            candidates.push(v1,d1);
//
//        }


        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
                if (nres < annk) {
                    faiss::maxheap_push(++nres, distances, results, dis, idx);
                } else if (dis < distances[0]) {
                    faiss::maxheap_replace_top(nres, distances, results, dis, idx);
                }
            candidates.push(idx, dis);
        };

        for (size_t j = 0; j < linkLists[v0]->bottom_connections; j++) {
            int v1 = v0_linkList[j];

            bool vget = vt.get(v1);
            vt.set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            if (counter == 4) {
                float dis[4];
                num_dco_expansion+=4;
                auto start = std::chrono::high_resolution_clock::now();
                disq.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);
                time_dco_expansion += chronoElapsedTime(start);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                counter = 0;
            }
        }

        for (size_t icnt = 0; icnt < counter; icnt++) {
            num_dco_expansion +=1;
            auto start = std::chrono::high_resolution_clock::now();
            float dis = disq(saved_j[icnt]);
            time_dco_expansion+= chronoElapsedTime(start);
            add_to_heap(saved_j[icnt], dis);
        }

        nstep++;
        delete[] v0_vector;
        if(nstep>dynamicParams.efSearch && nstep>annk) {
            break;
        }

    }
    //printf("nstep=%d\n", nstep);
    return nres;


}
/// Allow single-directional edge
void CANDY::DynamicTuneHNSW::cutEdgesBase(idx_t src) {
    auto node=linkLists[src];
    auto dists = node->distances[0];
    auto neighbors = node->neighbors[0];
    if(node->bottom_connections<dynamicParams.bottom_connections_lower_bound) {
        return;
    }
    int64_t prev_degree = node->bottom_connections;
    int64_t current_degree;
    float previous_distance_sum = 0.0;
    float current_distance_sum = 0.0;
    if(!dynamicParams.sparsePreference) {
        auto candidates = std::priority_queue<CandidateFarther>(); // no sparse preference->nearest to farthest
        for(int i=0; i<node->bottom_connections; i++) {
            candidates.emplace(dists[i], neighbors[i]);
            previous_distance_sum+=dists[i];
        }
        int i=0;
        while(i<dynamicParams.discardN){
            i++;
            candidates.pop();
        }
        i=0;
        while(candidates.empty()) {
            neighbors[i]=candidates.top().id;
            dists[i]=candidates.top().dist;
            current_distance_sum+=dists[i];
            candidates.pop();
            i++;
        }
        current_degree = i;
        node->bottom_connections=i;

    } else {
        auto candidates = std::priority_queue<CandidateCloser>(); // sparse preference->farthest to closer
        for(int i=0; i<node->bottom_connections; i++) {
            candidates.emplace(dists[i], neighbors[i]);
            previous_distance_sum+=dists[i];
        }
        int i=0;
        while(i<dynamicParams.discardN){
            i++;
            candidates.pop();
        }
        i=0;
        while(candidates.empty()) {
            neighbors[i]=candidates.top().id;
            dists[i]=candidates.top().dist;
            current_distance_sum+=dists[i];
            candidates.pop();
            i++;
        }
        current_degree = i;
        node->bottom_connections = i;
    }

    auto n=graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
    omp_set_lock(&state_lock);
    graphStates.global_stat.degree_variance = update_var_with_new_value(
        n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
        current_degree, prev_degree);
    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);

    auto current_distance_avg = current_distance_sum/(current_degree*1.0);
    auto previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
        n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
        current_distance_avg, previous_distance_avg);
    graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);
    omp_unset_lock(&state_lock);
}

void CANDY::DynamicTuneHNSW::getCluster(idx_t src, std::priority_queue<EdgeFarther>& edges, int expansion) {
    queue<std::pair<idx_t, int>> to_bfs;
    to_bfs.emplace(src,0);
    unordered_set<idx_t> visited;
    visited.insert(src);
    unordered_map<idx_t, int> steps_away;
    steps_away[src]=0;
    if(expansion==0){
        expansion = dynamicParams.clusterExpansionStep;
    }
    visited.insert(src);

    while(!to_bfs.empty() ) {
        auto current = to_bfs.front().first;
        auto current_step = to_bfs.front().second;
        if(current_step>expansion) {
            break;
        }
        to_bfs.pop();
        auto node = linkLists[current];
        for(int i=0; i<node->bottom_connections; i++) {
            if(visited.find(node->neighbors[0][i])==visited.end()) {
                visited.insert(node->neighbors[0][i]);
                steps_away[node->neighbors[0][i]]=current_step+1;
                to_bfs.push(std::pair<idx_t, int>(node->neighbors[0][i], current_step+1));

            }
            if(current_step+1<=expansion) {
                edges.push(EdgeFarther(node->distances[0][i], node->id, node->neighbors[0][i], i));
            }
        }
    }

}

void CANDY::DynamicTuneHNSW::getCluster(idx_t src, std::priority_queue<EdgeCloser>& edges, int expansion) {
    queue<std::pair<idx_t, int>> to_bfs;
    to_bfs.emplace(src,0);
    unordered_set<idx_t> visited;
    visited.insert(src);
    unordered_map<idx_t, int> steps_away;
    steps_away[src]=0;
    if(expansion==0) {
        expansion = dynamicParams.clusterExpansionStep;
    }
    visited.insert(src);

    while(!to_bfs.empty() ) {
        auto current = to_bfs.front().first;
        auto current_step = to_bfs.front().second;
        if(current_step>expansion) {
            break;
        }
        to_bfs.pop();
        auto node = linkLists[current];
        for(int i=0; i<node->bottom_connections; i++) {
            if(visited.find(node->neighbors[0][i])==visited.end()) {
                visited.insert(node->neighbors[0][i]);
                steps_away[node->neighbors[0][i]]=current_step+1;
                to_bfs.push(std::pair<idx_t, int>(node->neighbors[0][i], current_step+1));
                if(current_step+1<=expansion) {
                    edges.push(EdgeCloser(node->distances[0][i], node->id, node->neighbors[0][i], i));
                }
            }
        }
    }
}
// We just try to lower average neighbor distance in this case
void CANDY::DynamicTuneHNSW::cutEdgesWindow(WindowStates& window_states, int64_t mode) {
    ordered_map vertices;
    if(mode == 1) {
        vertices = window_states.newVertices;
    } else {
        vertices = window_states.oldVertices;
    }
    auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
    auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
        for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++) {
            auto node = it->second;

            //cutEdgesBase(node->id);

            std::priority_queue<EdgeCloser> edges;
            getCluster(node->id, edges, 0);
            int i=0;
            auto discardClucsterN = std::min(dynamicParams.discardClusterN*1.0, edges.size()*dynamicParams.discardClusterProp);
            while(i<(size_t)discardClucsterN && !edges.empty()) {
                auto to_cut = edges.top();

                edges.pop();
                auto src_node = linkLists[to_cut.src];
                if(src_node->bottom_connections*1.0<degree_avg-dynamicParams.degree_std_range*std_dev-1) {
                    continue;
                }
                auto prev_degree = src_node->bottom_connections;
                float previous_distance_sum = 0;
                float current_distance_sum = 0;
                for(int64_t j=0; j<to_cut.idx; j++) {
                    current_distance_sum+=src_node->distances[0][j];
                }
                //printf("cutting from %ld to %ld, previous degree %ld, current degree %ld\n",to_cut.src, to_cut.dest, prev_degree, src_node->bottom_connections-1 );
                for(int64_t j=to_cut.idx; j<src_node->bottom_connections-1; j++) {
                    src_node->neighbors[0][j]=src_node->neighbors[0][j+1];
                    src_node->distances[0][j]=src_node->distances[0][j+1];
                    current_distance_sum+=src_node->distances[0][j];
                }
                previous_distance_sum = current_distance_sum+to_cut.dist;
                src_node->bottom_connections--;
                auto current_degree = src_node->bottom_connections;


                auto n = graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
                graphStates.global_stat.degree_variance = update_var_with_new_value(
                    n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
                    current_degree, prev_degree);
                graphStates.global_stat.degree_sum+=(current_degree-prev_degree);

                auto current_distance_avg = current_degree==0?0:current_distance_sum/(current_degree*1.0);
                auto previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
                float old_variance = graphStates.global_stat.neighbor_distance_variance;
                auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                    n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
                    current_distance_avg, previous_distance_avg);
                graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);
                // if(graphStates.global_stat.neighbor_distance_variance<0 || graphStates.global_stat.degree_variance<0) {
                //     printf("edge cutted:%ld %ld %ld %f\n", to_cut.src, to_cut.dest,to_cut.idx,to_cut.dist);
                //     printf("current degree %ld, previous degree %ld\n", current_degree, prev_degree);
                //     printf("current avg %lf, previous avg %lf\n", current_distance_avg, previous_distance_avg);
                //     printf("old variance %lf, old sum %lf\n", old_variance, old_sum);
                //     exit(0);
                // }

                i++;
            }
        }
}


void CANDY::DynamicTuneHNSW::swapEdgesWindow(WindowStates& window_states, int64_t mode) {
    ordered_map vertices;
    if(mode == 1){
        vertices = window_states.newVertices;
    } else {
        vertices = window_states.oldVertices;
    }

    auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
    auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));

    for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++){
        auto vertex1 = it->second;
        auto vertex1_idx = vertex1->id;
        auto vertex1_distances = vertex1->distances[0];
        auto vertex1_neighbors = vertex1->neighbors[0];

        auto success = false;
        for(int i=0; i< nb_neighbors(0); i++){
            if(vertex1_neighbors[i]==-1){
                break;
            }
            if(vertex1_neighbors[i]==-2){
                continue;
            }
            auto vertex2_idx = vertex1_neighbors[i];
            auto dist12 = vertex1_distances[i];
            if(hasEdge(vertex1_idx, vertex2_idx) && hasEdge(vertex2_idx, vertex1_idx)){
                    improveEdges(vertex1_idx, vertex2_idx, dist12);
            }
        }

    }
}
/// called by swapEdgesWindow

bool CANDY::DynamicTuneHNSW::improveEdges(idx_t vertex1, idx_t vertex2, float dist12){
    // remove edge between vertex1 and vertex2 by adding temp self-loop
    changeEdge(vertex1, vertex2, vertex1, 0.f);
    changeEdge(vertex2, vertex1, vertex2, 0.f);
    if(improveEdges(vertex1, vertex2, vertex1, vertex2, dist12, 0)){
    } else {
        //should undo, to see if needed
        return false;
    }
    return true;
}

bool CANDY::DynamicTuneHNSW::improveEdges(idx_t vertex1, idx_t vertex2, idx_t vertex3, idx_t vertex4, float total_gain, const uint8_t steps){
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    DAGNN::VisitedTable vt(graphStates.global_stat.ntotal+graphStates.time_local_stat.ntotal);
    {
     auto vertex2_vector = get_vector(vertex2);
     std::vector<idx_t> entry_vertex_indices={vertex3, vertex4};
     std::vector<idx_t> top_list(25);
     std::vector<float> entry_vertex_distances(25);

     disq->set_query(vertex2_vector);
     search(*disq, 25, entry_vertex_indices.data(), entry_vertex_distances.data(), vt);

     // find a good vertex3 from vertex2's wanna-be nearest neighbors
     float best_gain = total_gain;
     float dist23 = std::numeric_limits<float>::lowest();
     float dist34 = std::numeric_limits<float>::lowest();

     for(int i=0; i<top_list.size(); i++){
         idx_t new_vertex3 = top_list[i];
         float dist2n3 = entry_vertex_distances[i];
         if(vertex1!=new_vertex3 && vertex2!=new_vertex3 && !hasEdge(vertex2, new_vertex3) && !hasEdge(vertex2, new_vertex3)){
             auto nv3 = linkLists[new_vertex3];
             auto nv3_neighbors = nv3->neighbors[0];
             auto nv3_distances=  nv3->distances[0];
             for(int j=0; j< nb_neighbors(0); j++){
                 if(nv3_neighbors[i]==-1){
                     break;
                 }
                 if(nv3_neighbors[i]==-2){
                     continue;
                 }
                 idx_t new_vertex4 = nv3_neighbors[i];
                 // link (2, newv3) and cut (newv3, newv4)
                 // gain = -dist(2, newv3) +
                 auto gain = total_gain - dist2n3+nv3_distances[j];
                 if(best_gain < gain && new_vertex4!=vertex2){
                     best_gain = gain;
                     vertex3 = new_vertex3;
                     vertex4 = new_vertex4;
                     dist23 = dist2n3;
                     dist34 = nv3_distances[j];
                 }
             }
         }
     }
     // no new_vertex3 found
     if(dist23 == std::numeric_limits<float>::lowest()){
         return false;
     }
     total_gain = total_gain - dist23+dist34;
     changeEdge(vertex2, vertex2, vertex3, dist23);
     changeEdge(vertex3, vertex4, vertex2, dist23);
     changeEdge(vertex4, vertex3, vertex4, 0.f);



     delete[] vertex2_vector;
    }
    // 2. try to connect vertex1 with vertex4
    {
        if(vertex1 == vertex4){
            std::vector<idx_t> entry_vertex_indices = {vertex2, vertex3};
            auto vertex4_vector = get_vector(vertex4);
            disq->set_query(vertex4_vector);
            std::vector<idx_t> top_list(25);
            std::vector<float> entry_vertex_distances(25);
            search(*disq, 25, entry_vertex_indices.data(), entry_vertex_distances.data(), vt);
            delete[] vertex4_vector;

            float best_gain = 0;
            idx_t best_selected_neighbor = 0;
            float best_old_neighbor_dist = 0;
            float best_new_neighbor_dist = 0;
            idx_t best_good_vertex = 0;
            float best_good_vertex_dist = 0;
            for(int i=0; i<top_list.size(); i++){
                auto good_vertex = top_list[i];
                if(vertex4!=good_vertex && !hasEdge(vertex4, good_vertex)&& !hasEdge(good_vertex, vertex4)){
                    auto good_vertex_dist = entry_vertex_distances[i];
                    auto gv = linkLists[good_vertex];
                    auto gv_neighbors = gv->neighbors[0];
                    auto gv_distances = gv->distances[0];
                    for(int j=0; j< nb_neighbors(0); j++){
                        if(gv_neighbors[j]==-1){
                            break;
                        }
                        if(gv_neighbors[j]==-2){
                            continue;
                        }
                        auto selected_neighbor = gv_neighbors[j];
                        if(vertex4!=selected_neighbor && !hasEdge(vertex4, selected_neighbor) && !hasEdge(selected_neighbor, vertex4)){
                            auto factor = 1;
                            auto old_neighbor_dist = gv_distances[j];
                            auto new_neighbor_dist = disq->symmetric_dis(vertex4, selected_neighbor);
                            // cut (good_vertex, selected_neighbor) and link(good_vertex, vector_4) and link(vector4, selected_neighbor)
                            float new_gain = total_gain + old_neighbor_dist*factor - good_vertex_dist-new_neighbor_dist;
                            if(best_gain<new_gain){
                                best_gain = new_gain;
                                best_selected_neighbor = selected_neighbor;
                                best_old_neighbor_dist = old_neighbor_dist;
                                best_new_neighbor_dist = new_neighbor_dist;
                                best_good_vertex = good_vertex;
                                best_good_vertex_dist = good_vertex_dist;
                            }
                        }
                    }
                }
            }
           if(best_gain > 0 ){
               // vertex4 == vertex1 so this has two self-loops
               changeEdge(vertex4, vertex4, best_good_vertex, best_good_vertex_dist);
               changeEdge(vertex4, vertex4, best_selected_neighbor, best_new_neighbor_dist);

               changeEdge(best_good_vertex, best_selected_neighbor, vertex4, best_good_vertex_dist);
               changeEdge(best_selected_neighbor, best_good_vertex, vertex4, best_new_neighbor_dist);
               return true;
           }
        } else {
            if(!hasEdge(vertex1, vertex4) && !hasEdge(vertex4, vertex1)){

                auto dist14 = disq->symmetric_dis(vertex1, vertex4);
                if(total_gain - dist14>0){
                    std::vector<idx_t> entry_vertex_indices={vertex2, vertex3};
                    /// TODO: check has path
                    changeEdge(vertex1, vertex1, vertex4, dist14);
                    changeEdge(vertex4, vertex4, vertex1, dist14);
                    return true;
                }

            }
        }
    }
    // Rest are directly from DEG original impl by keeping on
    // 3. maxtimum path length
    if(steps > 10){
        return false;
    }
    // 4. swap vertex1 and vertex4 each second round to give each a fair chance.
    if(steps%2 == 1){
        idx_t temp = vertex1;
        vertex1 = vertex4;
        vertex4 = vertex1;
    }
    // 5. early stop
    if(total_gain<0){
        return false;
    }
    return improveEdges(vertex1, vertex4, vertex2, vertex3,total_gain, steps+1);
}

void CANDY::DynamicTuneHNSW::linkEdgesWindow(WindowStates& window_states, int64_t mode) {
    ordered_map vertices;
    if(mode == 1) {
        vertices = window_states.newVertices;
    } else {
        vertices = window_states.oldVertices;
    }
    auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
    auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++) {
        auto node = it->second;
        // only processing those with too few edges
        if(node->bottom_connections*1.0>degree_avg+dynamicParams.degree_std_range*std_dev+1) {
            continue;
        }
        std::priority_queue<EdgeCloser> edges;
        getCluster(node->id, edges, 0);
        auto prev_degree = node->bottom_connections;
        float previous_distance_sum = 0.0;
        float current_distance_sum = 0.0;
        for(int64_t j=0; j<node->bottom_connections; j++) {
            previous_distance_sum+=node->distances[0][j];
        }

        current_distance_sum=previous_distance_sum;

        auto previous_distance_avg = previous_distance_sum/(prev_degree*1.0);
        auto current_distance_avg = previous_distance_avg;
        auto current_degree = prev_degree;

        auto node_vector = get_vector(node->id);
        while(current_degree < degree_avg+dynamicParams.degree_allow_range*std_dev && !edges.empty()) {
            auto to_add = edges.top();
            edges.pop();
            if(to_add.src==node->id) {
                //bad neighbor->cut
                for(int64_t j=to_add.idx; j<node->bottom_connections-1; j++) {
                    node->neighbors[0][j] = node->neighbors[0][j+1];
                    node->distances[0][j] = node->distances[0][j+1];
                }
                current_distance_sum -= to_add.dist;
                current_degree--;
                if(current_degree!=0) {
                    current_distance_avg = current_distance_sum/(current_degree*1.0);
                } else {
                    current_distance_avg =0;
                }
                node->bottom_connections = current_degree;
            } else if(to_add.src!=node->id && to_add.dest!=node->id) {
                // add outward edges
                auto nexus = linkLists[to_add.dest];
                for(int64_t i=0; i<nexus->bottom_connections; i++) {
                    //auto dest_vector = get_vector(nexus->neighbors[0][i]);
                    auto dist = disq->symmetric_dis(nexus->id, nexus->neighbors[0][i]);
                    if(dist<current_distance_avg) {
                        node->neighbors[0][node->bottom_connections] = nexus->neighbors[0][i];
                        node->distances[0][node->bottom_connections] = dist;

                        current_degree++;
                        node->bottom_connections=current_degree;
                        current_distance_sum += dist;
                        current_distance_avg = current_distance_sum/(current_degree*1.0);
                        if(current_degree >= degree_avg+dynamicParams.degree_allow_range*std_dev) {

                            //delete[] dest_vector;
                            break;
                        }
                    }

                    //delete[] dest_vector;
                }
            }
        }
        delete[] node_vector;
        auto n = graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
        graphStates.global_stat.degree_variance = update_var_with_new_value(
            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
            current_degree, prev_degree);
        graphStates.global_stat.degree_sum+=(current_degree-prev_degree);

        float old_variance = graphStates.global_stat.neighbor_distance_variance;
        auto old_sum = graphStates.global_stat.neighbor_distance_sum;
        graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
            n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
            current_distance_avg, previous_distance_avg);
        graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);

    }

}

void CANDY::DynamicTuneHNSW::liftClusterCenter(faiss::DistanceComputer& disq, idx_t src, DAGNN::VisitedTable& vt){
    auto node = linkLists[src];
    if(node->level!=0){
        printf("unable to lift this node up from level>0\n");
        return;
    }
    auto vector = get_vector(src);
    disq.set_query(vector);
    auto nearest = entry_points[0];
    auto dist_nearest = disq(nearest);
    // lift the node from 0 to level 1
    int64_t steps_taken = 0;

    node->level = 1;
    if(node->neighbors.size()!=2){
        auto l_neighbors = std::vector<idx_t>(nb_neighbors(1),-1);
        auto l_distances = std::vector<float>(nb_neighbors(1), -0.0);
        node->neighbors.push_back(l_neighbors);
        node->distances.push_back(l_distances);
    }

    for(size_t l = max_level; l>1; l--){
        greedy_insert_top(disq, l, nearest, dist_nearest, steps_taken);
    }

    std::priority_queue<CandidateCloser> candidates;
    greedy_insert_upper(disq, 1, nearest, dist_nearest, candidates, steps_taken);

    auto entry_first_level = CandidateCloser(dist_nearest, nearest);
    candidates.push(entry_first_level);
    link_from(disq, src, 1, nearest, dist_nearest,candidates, vt);
    // here no need to re-update steps_taken as it ends at level 1
    delete[] vector;
}

void CANDY::DynamicTuneHNSW::degradeNavigationPoint(faiss::DistanceComputer& disq, CANDY::DynamicTuneHNSW::idx_t src, DAGNN::VisitedTable& vt) {
    auto node = linkLists[src];
    if(node->level!=1){
        printf("unable to degrade node from level other than 1!\n");
    }

    // deletion of a point
    // simple link (p1,p2) for (p1, p) and (p,p2)
    int i=0;
    for(i=0; i< nb_neighbors(1); i++){
        if(node->neighbors[1][i]<0){
            break;
        }
    }
    int first_level_neighbor_nb = i;
    int left = 0;
    int right = first_level_neighbor_nb-1;
    while(left<right){
        auto left_node = linkLists[node->neighbors[1][left]];
        auto right_node = linkLists[node->neighbors[1][right]];
        auto dist = disq.symmetric_dis(node->neighbors[1][left], node->neighbors[1][right]);

        int left_idx = 0;
        int right_idx = 0;
        for(left_idx = 0; left_idx<nb_neighbors(1); left_idx++){
            if(left_node->neighbors[1][left_idx]<0){
                break;
            }
            if(left_node->neighbors[1][left_idx]==src){
                break;
            }
        }
        for(right_idx=0; right_idx<nb_neighbors(1); right_idx++){
            if(right_node->neighbors[1][right_idx]<0){
                break;
            }
            if(right_node->neighbors[1][right_idx]==src){
                break;
            }
        }
        left_node->neighbors[1][left_idx] = right_node->id;
        left_node->distances[1][left_idx] = dist;
        right_node->neighbors[1][right_idx] = left_node->id;
        right_node->distances[1][right_idx] = dist;
        left++;
        right--;
    }

    if(left==right){
        auto left_node = linkLists[node->neighbors[1][left]];
        int left_idx = 0;
        for(left_idx = 0; left_idx<nb_neighbors(1); left_idx++){
            if(left_node->neighbors[1][left_idx]<0){
                break;
            }
            if(left_node->neighbors[1][left_idx]==src){
                break;
            }
        }
        int i=left_idx;
        while(i<nb_neighbors(1)-1 ){
            if(left_node->neighbors[1][i]<0){
                break;
            }
            left_node->neighbors[1][i]=left_node->neighbors[1][i+1];
            i++;
        }
        left_node->neighbors[1][i]=-1;
    }


    node->level = 0;
    for(size_t i=0; i<nb_neighbors(1); i++){
        node->neighbors[1][i]=-1;
    }


}

void CANDY::DynamicTuneHNSW::hierarchyOptimizationDegradeWIndow(WindowStates& window_states){
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    std::vector<idx_t> degraded;
    auto vertices =window_states.hierarchyVertices;
    for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++){
        auto node = it->second;
        if(node->level!=1){
            continue;
        }
        DAGNN::VisitedTable vt(graphStates.global_stat.ntotal+graphStates.time_local_stat.ntotal);
        degradeNavigationPoint(*disq, node->id, vt);
        degraded.push_back(node->id);
    }
    for(size_t i=0; i<degraded.size(); i++){
        //window_states.hierarchyVertices.remove(degraded[i]);
    }


}

void CANDY::DynamicTuneHNSW::hierarchyOptimizationLiftWIndow(WindowStates& window_states){
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    std::vector<idx_t> lifted;
    auto vertices =window_states.hierarchyVertices;
    for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++){
        auto node = it->second;
        if(node->level!=0){
            continue;
        }
        DAGNN::VisitedTable vt(graphStates.global_stat.ntotal+graphStates.time_local_stat.ntotal);
        liftClusterCenter(*disq, node->id, vt);
        lifted.push_back(node->id);
    }
    for(size_t i=0; i<lifted.size(); i++){
       //window_states.hierarchyVertices.remove(lifted[i]);
    }

}

void CANDY::DynamicTuneHNSW::backtrackCandidate(faiss::DistanceComputer& disq, idx_t src, DAGNN::VisitedTable& vt){
    auto entry_point = entry_points[0];
    auto entry_vector = get_vector(entry_point);
    disq.set_query(entry_vector);
    auto backtracking = src;
    auto dist_backtracking = disq(src);
    int step_taken = 0;
    std::priority_queue<CandidateCloser> candidates;
    size_t nb_neighbor_level = nb_neighbors(0);

    while(step_taken < dynamicParams.max_backtrack_steps){
        idx_t prev_backtracking = backtracking;
        auto node = linkLists[backtracking];
        for(size_t i=0; i<nb_neighbor_level; i++){
            auto visiting = node->neighbors[0][i];
            if(visiting<0){
                break;
            }
            auto dist = disq(visiting);

            if(dist<dist_backtracking){
                backtracking = visiting;
                dist_backtracking = dist;
            }
        }
        if(backtracking == prev_backtracking){
            break;
        }
        step_taken++;
    }

    // now we've arrived at the wanted place near the entry, we would add links around to src

    auto entry_base_level = CandidateCloser(dist_backtracking, backtracking);
    candidates.push(entry_base_level);
    /// to add some neighbors near entry to src with disq's query as entry
    link_from_base(disq, src, backtracking, dist_backtracking, candidates, vt);

}

void CANDY::DynamicTuneHNSW::linkClusters(faiss::DistanceComputer& disq, idx_t src, idx_t dest, DAGNN::VisitedTable& vt){
    if(src==dest){
        //printf("Do not attempt to link within one cluster for this set of actions!\n");
        return;
    }

    auto edges_src =std::priority_queue<EdgeCloser>();
    auto edges_dest = std::priority_queue<EdgeCloser>();

    getCluster(src, edges_src, 0);
    getCluster(dest, edges_dest, 0);

    //Replace long links within cluster with an bi-directional edge to another cluster to allow possible fast transportation
    size_t count = 0;
    size_t overlapping_count = 0;
    while(!edges_src.empty() && !edges_dest.empty() && count < dynamicParams.nb_navigation_paths){
        auto edge_src = edges_src.top();
        auto edge_dest = edges_dest.top();
        edges_src.pop();
        edges_dest.pop();

        // Plan1: (n1,n2) & (n3, n4) -> (n1,n4) & (n3,n2) with some conditions
        // Plan2: (n1,n2) & (n3,n4) -> (n1,n3) & (n2, n4)
        auto node_1_id = edge_src.src;
        auto node_2_id = edge_src.dest;
        auto node_3_id = edge_dest.src;
        auto node_4_id = edge_dest.dest;

        double node_1_sum = 0.0;
        double node_2_sum = 0.0;
        double node_3_sum = 0.0;
        double node_4_sum = 0.0;


        auto idx_1 = edge_src.idx;
        auto idx_3 = edge_dest.idx;
        if(node_1_id == node_3_id || node_1_id == node_4_id || node_2_id == node_3_id || node_2_id == node_4_id){
            // overlapping so continue
            overlapping_count++;
            if(overlapping_count>16){
                // likely to be in the same cluster
                break;
            }
            continue;
        }
        auto node_1 = linkLists[node_1_id];
        bool plan_1 = true;
        bool plan_2 = true;
        for(int i=0; i< nb_neighbors(0); i++){
            if(node_3_id == node_1->neighbors[0][i]) {
                plan_2 = false;
            }
            if(node_4_id == node_1->neighbors[0][i]){
                plan_1 = false;
            }
            if(-1 == node_1->neighbors[0][i]){
                break;
            }
            node_1_sum += node_1->distances[0][i];
        }
        if(!plan_1 && !plan_2){
            count++;
            continue;
        }

        auto node_3 = linkLists[node_3_id];
        for(int i=0; i<nb_neighbors(0); i++){
            if(node_1_id == node_3->neighbors[0][i]){
                plan_2 = false;
            }
            if(node_2_id == node_3->neighbors[0][i]){
                plan_1 = false;
            }
            if(-1==node_3->neighbors[0][i]){
                break;
            }
            node_3_sum+=node_3->distances[0][i];
        }
        if(!plan_1 && !plan_2){
            count++;
            continue;
        }
        auto idx_2 = -1;
        auto node_2 = linkLists[node_2_id];
        for(int i=0; i<nb_neighbors(0); i++){
            if(node_4_id == node_2->neighbors[0][i]){
                plan_2 = false;
            }
            if(node_3_id == node_2->neighbors[0][i]){
                plan_1 = false;
            }
            if(node_1_id == node_2->neighbors[0][i]){
                idx_2 = i;
            }
            if(-1 == node_2->neighbors[0][i] && idx_2==-1){
                idx_2 = i;
                break;
            }
            node_2_sum+=node_2->distances[0][i];
        }
        if((!plan_1 && !plan_2) || (idx_2==-1)){
            count++;
            continue;
        }
        auto node_4 = linkLists[node_4_id];
        auto idx_4 = -1;
        for(int i=0; i<nb_neighbors(0); i++){
            if(node_2_id == node_4->neighbors[0][i]){
                plan_2 = false;
            }
            if(node_1_id == node_4->neighbors[0][i]){
                plan_1 = false;
            }
            if(node_3_id == node_4->neighbors[0][i]){
                idx_4 = i;
            }
            if(-1 == node_4->neighbors[0][i] && idx_4==-1){
                idx_4 = i;
                break;
            }
            node_4_sum+=node_4->distances[0][i];
        }
        if((!plan_1 && !plan_2)||(idx_4==-1)){
            count++;
            continue;
        }

        auto length14 = disq.symmetric_dis(node_1_id, node_4_id);
        auto length23 = disq.symmetric_dis(node_2_id, node_3_id);
        auto length13 = disq.symmetric_dis(node_1_id, node_3_id);
        auto length24 = disq.symmetric_dis(node_2_id, node_4_id);


        if(plan_1 && plan_2){
            // choose a plan that brings less total distance
            if(length14+length23 >= length13+length24){
                plan_1 = false;
            } else {
                plan_2 = false;
            }
        }
        auto n = graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal;
        if(plan_1) {
            //printf("executing plan 1:\n");
            // (n1,n2)(n3,n4)->(n1,n4)(n2,n3)
            //node1
            {
                auto previous_distance = node_1->distances[0][idx_1];
                node_1->neighbors[0][idx_1] = node_4_id;
                node_1->distances[0][idx_1] = length14;
                auto previous_distance_avg = node_1_sum / (node_1->bottom_connections * 1.0);
                auto current_distance_avg =
                        (node_1_sum - previous_distance + length14) / (node_1->bottom_connections * 1.0);
                float old_variance = graphStates.global_stat.neighbor_distance_variance;
                auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                        n, graphStates.global_stat.neighbor_distance_variance,
                        graphStates.global_stat.neighbor_distance_sum,
                        current_distance_avg, previous_distance_avg);
                graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
            }
            //node3
            {
                auto previous_distance = node_3->distances[0][idx_3];
                node_3->neighbors[0][idx_3]=node_2_id;
                node_3->distances[0][idx_3] = length23;
                auto previous_distance_avg = node_3_sum/(node_3->bottom_connections*1.0);
                auto current_distance_avg = (node_3_sum-previous_distance+length23)/(node_3->bottom_connections*1.0);
                float old_variance = graphStates.global_stat.neighbor_distance_variance;
                auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                        n, graphStates.global_stat.neighbor_distance_variance,
                        graphStates.global_stat.neighbor_distance_sum,
                        current_distance_avg, previous_distance_avg);
                graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
            }
            //node2
            {
                if(idx_2<node_2->bottom_connections){
                    auto previous_distance = node_2->distances[0][idx_2];
                    node_2->neighbors[0][idx_2] = node_3_id;
                    node_2->distances[0][idx_2] = length23;
                    auto previous_distance_avg = node_2_sum/(node_2->bottom_connections*1.0);
                    auto current_distance_avg = (node_2_sum-previous_distance_avg+length23)/(node_2->bottom_connections*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance,
                            graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
                } else {
                    auto previous_distance = 0;
                    node_2->neighbors[0][idx_2]=node_3_id;
                    node_2->distances[0][idx_2]=length23;
                    auto prev_degree = node_2->bottom_connections;
                    node_2->bottom_connections++;
                    auto current_degree = node_2->bottom_connections;
                    graphStates.global_stat.degree_variance = update_var_with_new_value(
                            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
                            current_degree, prev_degree);
                    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);
                    auto previous_distance_avg = node_2_sum/(prev_degree*1.0);
                    auto current_distance_avg = (node_2_sum + length23)/(current_degree*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);

                }
            }
            //node4
            {
                if(idx_4<node_4->bottom_connections){
                    auto previous_distance = node_4->distances[0][idx_4];
                    node_4->neighbors[0][idx_4] = node_1_id;
                    node_4->distances[0][idx_4] = length14;
                    auto previous_distance_avg = node_4_sum/(node_4->bottom_connections*1.0);
                    auto current_distance_avg = (node_4_sum-previous_distance_avg+length14)/(node_4->bottom_connections*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance,
                            graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
                } else {
                    auto previous_distance = 0;
                    node_4->neighbors[0][idx_4]=node_1_id;
                    node_4->distances[0][idx_4]=length14;
                    auto prev_degree = node_4->bottom_connections;
                    node_4->bottom_connections++;
                    auto current_degree = node_4->bottom_connections;
                    graphStates.global_stat.degree_variance = update_var_with_new_value(
                            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
                            current_degree, prev_degree);
                    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);
                    auto previous_distance_avg = node_4_sum/(prev_degree*1.0);
                    auto current_distance_avg = (node_4_sum + length14)/(current_degree*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);
                }
            }
        } else if(plan_2){
            // (n1,n2)(n3,n4)->(n1,n3)&(n2,n4)
            //printf("executing plan 2\n");
            //node1
            {
                auto previous_distance = node_1->distances[0][idx_1];
                node_1->neighbors[0][idx_1] = node_3_id;
                node_1->distances[0][idx_1] = length13;
                auto previous_distance_avg = node_1_sum / (node_1->bottom_connections * 1.0);
                auto current_distance_avg =
                        (node_1_sum - previous_distance + length13) / (node_1->bottom_connections * 1.0);
                float old_variance = graphStates.global_stat.neighbor_distance_variance;
                auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                        n, graphStates.global_stat.neighbor_distance_variance,
                        graphStates.global_stat.neighbor_distance_sum,
                        current_distance_avg, previous_distance_avg);
                graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
            }
            //node3
            {
                auto previous_distance = node_3->distances[0][idx_3];
                node_3->neighbors[0][idx_3]=node_1_id;
                node_3->distances[0][idx_3] = length13;
                auto previous_distance_avg = node_3_sum/(node_3->bottom_connections*1.0);
                auto current_distance_avg = (node_3_sum-previous_distance+length13)/(node_3->bottom_connections*1.0);
                float old_variance = graphStates.global_stat.neighbor_distance_variance;
                auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                        n, graphStates.global_stat.neighbor_distance_variance,
                        graphStates.global_stat.neighbor_distance_sum,
                        current_distance_avg, previous_distance_avg);
                graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
            }
            //node2
            {
                if(idx_2<node_2->bottom_connections){
                    auto previous_distance = node_2->distances[0][idx_2];
                    node_2->neighbors[0][idx_2] = node_4_id;
                    node_2->distances[0][idx_2] = length24;
                    auto previous_distance_avg = node_2_sum/(node_2->bottom_connections*1.0);
                    auto current_distance_avg = (node_2_sum-previous_distance_avg+length24)/(node_2->bottom_connections*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance,
                            graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
                } else {
                    auto previous_distance = 0;
                    node_2->neighbors[0][idx_2]=node_4_id;
                    node_2->distances[0][idx_2]=length24;
                    auto prev_degree = node_2->bottom_connections;
                    node_2->bottom_connections++;
                    auto current_degree = node_2->bottom_connections;
                    graphStates.global_stat.degree_variance = update_var_with_new_value(
                            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
                            current_degree, prev_degree);
                    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);
                    auto previous_distance_avg = node_2_sum/(prev_degree*1.0);
                    auto current_distance_avg = (node_2_sum + length24)/(current_degree*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);

                }
            }
            //node4
            {
                if(idx_4<node_4->bottom_connections){
                    auto previous_distance = node_4->distances[0][idx_4];
                    node_4->neighbors[0][idx_4] = node_2_id;
                    node_4->distances[0][idx_4] = length24;
                    auto previous_distance_avg = node_4_sum/(node_4->bottom_connections*1.0);
                    auto current_distance_avg = (node_4_sum-previous_distance_avg+length24)/(node_4->bottom_connections*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance,
                            graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum += (current_distance_avg - previous_distance_avg);
                } else {
                    auto previous_distance = 0;
                    node_4->neighbors[0][idx_4]=node_2_id;
                    node_4->distances[0][idx_4]=length24;
                    auto prev_degree = node_4->bottom_connections;
                    node_4->bottom_connections++;
                    auto current_degree = node_4->bottom_connections;
                    graphStates.global_stat.degree_variance = update_var_with_new_value(
                            n, graphStates.global_stat.degree_variance,graphStates.global_stat.degree_sum,
                            current_degree, prev_degree);
                    graphStates.global_stat.degree_sum+=(current_degree-prev_degree);
                    auto previous_distance_avg = node_4_sum/(prev_degree*1.0);
                    auto current_distance_avg = (node_4_sum + length24)/(current_degree*1.0);
                    float old_variance = graphStates.global_stat.neighbor_distance_variance;
                    auto old_sum = graphStates.global_stat.neighbor_distance_sum;
                    graphStates.global_stat.neighbor_distance_variance = update_var_with_new_value(
                            n, graphStates.global_stat.neighbor_distance_variance, graphStates.global_stat.neighbor_distance_sum,
                            current_distance_avg, previous_distance_avg);
                    graphStates.global_stat.neighbor_distance_sum+=(current_distance_avg-previous_distance_avg);
                }
            }
        } else{
             //printf("no plan to execute\n");

        }



        count++;
    }



}

void CANDY::DynamicTuneHNSW::navigationBacktrackWindow(WindowStates& window_states){
    auto vertices =window_states.oldVertices;
    auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
    auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    for(auto it = vertices.data_map.begin(); it!=vertices.data_map.end(); it++){
        auto node = it->second;
        if(node->bottom_connections*1.0>degree_avg+dynamicParams.degree_std_range*std_dev+1) {
            continue;
        }
        DAGNN::VisitedTable vt(graphStates.global_stat.ntotal+graphStates.time_local_stat.ntotal);
        backtrackCandidate(*disq, node->id, vt);
    }
}

void CANDY::DynamicTuneHNSW::linkClusterWindow(CANDY::DynamicTuneHNSW::WindowStates &window_states) {
    std::unique_ptr<faiss::DistanceComputer> disq(
            storage_distance_computer(storage));
    auto degree_avg = graphStates.global_stat.degree_sum/(graphStates.global_stat.ntotal*1.0);
    auto std_dev= std::sqrt(graphStates.global_stat.degree_variance);
    auto it1 = window_states.oldVertices.data_map.begin();
    auto it2 = window_states.newVertices.data_map.begin();
    while(it1!=window_states.oldVertices.data_map.end() && it2!=window_states.newVertices.data_map.end()){
        auto node1 = it1->second;
        auto node2 = it2->second;
        if(node1->bottom_connections<degree_avg-dynamicParams.degree_allow_range*std_dev-1 || node2->bottom_connections<degree_avg-dynamicParams.degree_allow_range*std_dev-1) {
            DAGNN::VisitedTable vt(graphStates.global_stat.ntotal + graphStates.time_local_stat.ntotal);
            linkClusters(*disq, it1->first, it2->first, vt);
        }
        it1++;
        it2++;
    }
}

void CANDY::DynamicTuneHNSW::deleteVector(CANDY::DynamicTuneHNSW::idx_t n, float *x) {
    // global search for deleted node
}

void CANDY::DynamicTuneHNSW::deleteVectorByIndex(const std::vector<idx_t>& idx) {

    if(delete_mode==0) {
        // simple tombstone
        for (size_t i = 0; i < idx.size(); i++) {
            deleteLists[idx[i]] = true;
        }
    } else if (delete_mode == 1){
        // global reconnect, need to re-insert to_delete's neighbors
        for(size_t i=0; i<idx.size(); i++){
            auto id = idx[i];
            deleteLists[id]=true;
            auto to_delete = linkLists[id];
            // re-greedy-insert to_delete's neighbors, only for base level, we maintain the navigation point on upper levels
            auto neighbors_base = to_delete->neighbors[0];
            for(size_t n=0; n< nb_neighbors(0); n++){
                auto nid = neighbors_base[n];
                if(nid<0){
                    break;
                }

                std::unique_ptr<faiss::DistanceComputer> disq(
                        storage_distance_computer(storage));
                DAGNN::VisitedTable vt(storage->ntotal);
                auto node = linkLists[nid];
                auto vector = get_vector(nid);
                disq->set_query(vector);
                auto nearest = entry_points[0];
                auto dist_nearest = (*disq)(nearest);
                std::priority_queue<CandidateCloser> candidates;
                int64_t steps_taken = 0;
                greedy_insert_base(*disq, nearest, dist_nearest, candidates, steps_taken);

                auto entry_base_level = CandidateCloser(dist_nearest, nearest);
                candidates.push(entry_base_level);
                link_from_base(*disq, node->id,  nearest, dist_nearest, candidates, vt);

                delete[] vector;
            }
        }
    } else if (delete_mode == 2){
        // local reconnect
        // global reconnect, need to re-insert to_delete's neighbors
        for(size_t i=0; i<idx.size(); i++){
            auto id = idx[i];
            deleteLists[id]=true;
            auto to_delete = linkLists[id];
            // local connect, re-insert using deleted vectexas entry, only for base level, we maintain the navigation point on upper levels
            auto neighbors_base = to_delete->neighbors[0];
            for(size_t n=0; n< nb_neighbors(0); n++){
                auto nid = neighbors_base[n];
                if(nid<0){
                    break;
                }

                std::unique_ptr<faiss::DistanceComputer> disq(
                        storage_distance_computer(storage));
                DAGNN::VisitedTable vt(storage->ntotal);
                auto node = linkLists[nid];
                auto vector = get_vector(nid);
                disq->set_query(vector);
                auto nearest = entry_points[0];
                auto dist_nearest = (*disq)(nearest);
                std::priority_queue<CandidateCloser> candidates;
                int64_t steps_taken = 0;
                greedy_insert_base(*disq, nearest, dist_nearest, candidates, steps_taken);

                auto entry_base_level = CandidateCloser(dist_nearest, nearest);
                candidates.push(entry_base_level);
                link_from_base(*disq, node->id,  nearest, dist_nearest, candidates, vt);

                delete[] vector;
            }
        }
    }

}

bool CANDY::DynamicTuneHNSW::performAction(const size_t action_num) {
	graphStates.window_states.last_action = action_num;
    switch(action_num){
        case do_nothing:
            //printf("do nothing\n");
            break;
        case bad_link_cut_old:
            //printf("bad link cut old\n");
            cutEdgesWindow(graphStates.window_states, 0);
            break;
        case bad_link_cut_new:
            //printf("bad link cut new\n");
            cutEdgesWindow(graphStates.window_states, 1);
            break;
        case outwards_link_old:
            //printf("outwards link old\n");
            linkEdgesWindow(graphStates.window_states, 0);
            break;
        case outwards_link_new:
            //printf("outwards link new\n");
            //linkEdgesWindow(graphStates.window_states,1);
            break;
//        case DEG_refine_old:
//            printf("DEG old\n");
//            swapEdgesWindow(graphStates.window_states, 0);
//            break;
//        case DEG_refine_new:
//            printf("DEG new\n");
//            swapEdgesWindow(graphStates.window_states,1);
//            break;
        case backtrack_candidate:
            //printf("backtrack candidate\n");
            navigationBacktrackWindow(graphStates.window_states);
            break;
        case intercluster_link:
            //printf("intercluster link\n");
            linkClusterWindow(graphStates.window_states);

            break;
        case lift_cluster_center:
            //printf("lift cluster center\n");
            hierarchyOptimizationLiftWIndow(graphStates.window_states);
            break;
        case lower_navigation_point:
            //printf("lower navigation point\n");
            hierarchyOptimizationDegradeWIndow(graphStates.window_states);
            break;
        case increase_rng_alpha:
            printf("increase rng alpha\n");
            if(dynamicParams.rng_alpha<1.5) {
                dynamicParams.rng_alpha += 0.05;
            }
            break;
        case decrease_rng_alpha:
            if(dynamicParams.rng_alpha>1){
                dynamicParams.rng_alpha-=0.05;
            }
            break;
        case increase_cluster_expansion:
            if(dynamicParams.clusterExpansionStep<4){
                dynamicParams.clusterExpansionStep+=1;
            }
            break;
        case decrease_cluster_expansion:
            if(dynamicParams.clusterExpansionStep>1){
                dynamicParams.clusterExpansionStep-=1;
            }
            break;
        case increase_cluster_innerconnection_threshold:
            if(dynamicParams.clusterInnerConnectionThreshold <0.75){
                dynamicParams.clusterExpansionStep+=0.05;
            }
            break;
        case decrease_cluster_innerconnection_threshold:
            if(dynamicParams.clusterInnerConnectionThreshold>0.15){
                dynamicParams.clusterExpansionStep-=0.05;
            }
            break;
        case increase_optimisticN:
            if(dynamicParams.optimisticN <96){
                dynamicParams.optimisticN+=8;
            }
            break;
        case decrease_optimisticN:
            if(dynamicParams.optimisticN>0){
                dynamicParams.optimisticN-=8;
            }
            break;
        case increase_discardN:
            if(dynamicParams.discardN<8){
                dynamicParams.discardN+=1;
            }
            break;
        case decrease_discardN:
            if(dynamicParams.discardN>0){
                dynamicParams.discardN-=1;
            }
            break;
        case increase_discardClusterProp:
            if(dynamicParams.discardClusterProp<0.75){
                dynamicParams.discardClusterProp+=0.05;
            }
            break;
        case decrease_discardClusterProp:
            if(dynamicParams.discardClusterProp>0.1){
                dynamicParams.discardClusterProp-=0.05;
            }
            break;
        case increase_discardClusterN:
            if(dynamicParams.discardClusterN < 256){
                dynamicParams.discardClusterProp+=16;
            }
            break;
        case decrease_discardClusterN:
            if(dynamicParams.discardClusterN>16){
                dynamicParams.discardClusterProp-=16;
            }
            break;
        case increase_expansionConstruction:
            if(dynamicParams.efConstruction < 240){
                dynamicParams.efConstruction+=10;
            }
            break;
        case decrease_expansionConstruction:
            if(dynamicParams.efConstruction>20){
                dynamicParams.efConstruction-=10;
            }
            break;
        case increase_degree_std_range:
            if(dynamicParams.degree_std_range<2.25){
                dynamicParams.degree_std_range+=0.125;
            }
            break;
        case decrease_degree_std_range:
            if(dynamicParams.degree_std_range>0.5){
                dynamicParams.degree_std_range-=0.125;
            }
            break;
        case increase_degree_allow_range:
            if(dynamicParams.degree_allow_range<0.75){
                dynamicParams.degree_allow_range+=0.125;
            }
            break;
        case decrease_degree_allow_range:
            if(dynamicParams.degree_allow_range>0.125){
                dynamicParams.degree_allow_range-=0.125;
            }
            break;
        case increase_sparsePreference:
            dynamicParams.sparsePreference=1;
            break;
        case decrease_sparsePreference:
            dynamicParams.sparsePreference=0;
            break;
        case increase_neighborDistanceThreshold:
            if(dynamicParams.neighborDistanceThreshold<0.75){
                dynamicParams.neighborDistanceThreshold+=0.125;
            }
            break;
        case decrease_neighborDistanceThreshold:
            if(dynamicParams.neighborDistanceThreshold>0.125){
                dynamicParams.neighborDistanceThreshold-=0.125;
            }
            break;
        case increasae_max_backtrack_steps:
            if(dynamicParams.max_backtrack_steps<40){
                dynamicParams.max_backtrack_steps+=2;
            }
            break;
        case decrease_max_backtrack_steps:
            if(dynamicParams.max_backtrack_steps>8){
                dynamicParams.max_backtrack_steps-=2;
            }
            break;
        case increase_steps_above_avg:
            if(dynamicParams.steps_above_avg < 150){
                dynamicParams.steps_above_avg+=10;
            }
            break;
        case decrease_steps_above_avg:
            if(dynamicParams.steps_above_avg>20){
                dynamicParams.steps_above_avg-=10;
            }
            break;
        case increase_steps_above_max:
            if(dynamicParams.steps_above_max < 100){
                dynamicParams.steps_above_max+=5;
            }
            break;
        case decrease_steps_above_max:
            if(dynamicParams.steps_above_max>5){
                dynamicParams.steps_above_max-=5;
            }
            break;
        case increase_nb_navigation_paths:
            if(dynamicParams.nb_navigation_paths < 96){
                dynamicParams.nb_navigation_paths+=8;
            }
            break;
        case decrease_nb_navigation_paths:
            if(dynamicParams.nb_navigation_paths>8){
                dynamicParams.nb_navigation_paths-=8;
            }
            break;
        case increase_expiration_timestamp:
            if(dynamicParams.expiration_timestamp<2500){
                dynamicParams.expiration_timestamp+=50;
            }
            break;
        case decrease_expiration_timestamp:
            if(dynamicParams.expiration_timestamp>300){
                dynamicParams.expiration_timestamp-=50;
            }
            break;
        case increase_degree_lift_range:
            if(dynamicParams.degree_lift_range < 2.5){
                dynamicParams.degree_lift_range+=0.125;
            }
            break;
        case decrease_degree_lift_range:
            if(dynamicParams.degree_lift_range>1.0){
                dynamicParams.degree_lift_range-=0.125;
            }
            break;
        default:
            return false;
    }
    //printf("action complete!\n");
    return true;
}
