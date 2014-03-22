#include <iostream>
#include <vector>
#include <chrono>

#include "componentcontainer.h"
#include "componentmodels.h"
#include "relation.h"

using namespace irm; 

int main()
{

    std::chrono::time_point<std::chrono::system_clock> start, end;

    const int ENTITY_N = 1000; 
    const int GROUPS = 32; 

    rng_t rng; 

    std::string data(ENTITY_N * ENTITY_N*sizeof(LogisticDistance::value_t), 0); 
    std::vector<size_t> shape = {ENTITY_N, ENTITY_N}; 
    std::string observed(ENTITY_N * ENTITY_N, 1); 
    ComponentContainer<LogisticDistance> cc_bb(data, shape, observed); 

    axesdef_t axes_def = {0, 0}; 
    domainsizes_t domainsizes = {ENTITY_N}; 

    boost::python::dict hps; 
    // hps["lambda_hp"] = 1.0; 
    // hps["mu_hp"] = 1.0; 
    // hps["p_min"] = 0.1; 
    // hps["p_max"] = 0.9; 
    //cc_bb.set_hps(hps); 
    Relation rel(axes_def, domainsizes, &cc_bb); 

    std::vector<groupid_t> assignments(ENTITY_N); 
    std::vector<groupid_t> groups(GROUPS +1); 

    // create the groups and add all the entities to them
    for(int g = 0; g < GROUPS; ++g) { 
        auto gid = rel.create_group(0, rng); 
        groups[g] = gid; 
        for(int i = 0; i < ENTITY_N; ++i) { 
            if(i % GROUPS == g) {
                rel.add_entity_to_group(0, gid, i); 
                assignments[i] = gid; 
            }
        }
    }

    float totalscore = 0; 
    // fake gibbs scan
    const int ITERS = 50; 
    for(int iter = 0; iter < ITERS; ++iter) { 
        start = std::chrono::system_clock::now();

        for(int ei = 0; ei < ENTITY_N; ++ei) { 
            auto init_gid = assignments[ei]; 
            rel.remove_entity_from_group(0, init_gid, ei); 
            // this group isn't empty 

            // create ephemeral group
            groups[GROUPS] = rel.create_group(0, rng); 

            // now compute post pred
            for(auto gid : groups) { 
                totalscore += rel.post_pred(0, gid, ei); 
            }
            
            rel.add_entity_to_group(0, init_gid, ei); 
            rel.delete_group(0, groups[GROUPS]); 

        }
        end = std::chrono::system_clock::now();
        auto dur = end - start; 

        std::cout << "iter " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()/1000.
                  << " sec"  << std::endl; 
        rel.get_datapoints_per_group(); 
    }
    std::cout << "done" << std::endl; 
    

}
