#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "data.h"
#include "graph_index/navigable_small_world.h"
#include "graph_index/hierarchical_navigable_small_world.h"

using namespace std;

int main(int argc,char** argv){

    //required variables from external input
    string base_path = argv[1];
    string graph_type = argv[2];
    int num_of_candidates = atoi(argv[3]);
    int num_of_initial_neighbors = atoi(argv[4]);

    cout << "Load data points..." << endl << endl;
    Data* points = new Data(base_path);

    GraphWrapper* graph;
    if (graph_type == "nsw") {
        graph = new NavigableSmallWorldGraphWithFixedDegree(points);
    } else if (graph_type == "hnsw") {
        graph = new HierarchicalNavigableSmallWorld(points);
    }
	
    cout << "Construct proximity graph " << graph_type << "..." << endl << endl;
    graph->Establishment(num_of_initial_neighbors, num_of_candidates);
    
    cout << "Save proximity graph " << graph_type << "..." << endl << endl;
    string graph_name = base_path+"_"+argv[3]+"_"+argv[4]+"."+graph_type;
    graph->Dump(graph_name);
    cout << "Done" << endl;
    
    return 0;
}
