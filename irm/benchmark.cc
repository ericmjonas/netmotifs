#include <iostream>
#include <vector>

#include "componentcontainer.h"
#include "componentmodels.h"
#include "relation.h"


using namespace irm; 

int main()
{

    const int ENTITY_N = 512; 
    const int GROUPS = 32; 

    rng_t rng; 

    std::string data(ENTITY_N * ENTITY_N*sizeof(LogisticDistance::value_t), 0); 
    std::vector<size_t> shape = {ENTITY_N, ENTITY_N}; 
    ComponentContainer<LogisticDistance> cc_bb(data, shape); 
    std::cout << "HERE" << std::endl; 
    axesdef_t axes_def = {0, 0}; 
    domainsizes_t domainsizes = {ENTITY_N}; 

    boost::python::dict hps; 
    hps["lambda_hp"] = 1.0; 
    hps["mu_hp"] = 1.0; 
    hps["p_min"] = 0.1; 
    hps["p_max"] = 0.9; 
    hps.has_key(boost::python::str("force_lbah")); 
    std::cout << "setting hps" << std::endl; 
    cc_bb.set_hps(hps); 
    std::cout << "HERE" << std::endl; 
    Relation rel(axes_def, domainsizes, &cc_bb); 

    std::vector<groupid_t> assignments(ENTITY_N); 
    std::vector<groupid_t> groups(GROUPS +1); 
    std::cout << "HERE2" << std::endl; 
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
    const int ITERS = 10; 
    for(int iter = 0; iter < ITERS; ++iter) { 
        std::cout << "iter " << iter << std::endl; 
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
    }
    std::cout << "done" << std::endl; 
    

}
