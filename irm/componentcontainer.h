#include <iostream>
#include <map>
#include <inttypes.h>
#include "util.h"

namespace irm { 

class IComponentContainer {
public:
    virtual size_t dpcount() = 0; 
    virtual float total_score() = 0; 
    virtual void create_component(group_coords_t group_coords); 
    virtual void delete_component(group_coords_t group_coords); 

    virtual float post_pred(group_coords_t group_coords, dppos_t dp_pos); 
    virtual void add_dp(group_coords_t group_coords, dppos_t dp_pos); 
    virtual void rem_dp(group_coords_t group_coords, dppos_t dp_pos) ; 


}; 
   
template<typename CM>
class  ComponentContainer : public IComponentContainer{ 
    
    struct sswrapper_t { 
        size_t count; 
        typename CM::suffstats_t ss; 
    }; 

    typedef uint64_t group_hash_t; 

public:
    ComponentContainer(int ndim, char * data, 
                       size_t data_shape[] ) : 
        NDIM_(ndim), 
        pdata_(static_cast<typename CM::value_t *>(data))
    {
        for(int i = 0; i < NDIM_; i++) { 
            data_shape_[i] = data_shape[i]; 
        }

        
    }
    
    void create_component(group_coords_t group_coords) {
        group_hash_t gp = hash_coords(group_coords); 
        sswrapper_t * ssw = new sswrapper_t; 
        CM::ss_init(&(ssw->ss), &hps_); 
        ssw->count = 0; 
        components_.insert(std::make_pair(gp, ssw)); 
    }

    void delete_component(group_coords_t group_coords) {
        group_hash_t gp = hash_coords(group_coords); 
        typename components_t::iterator i = components_.find(gp); 
        delete i->second; 
        components_.erase(i); 
    }

    float total_score() {
        typename components_t::iterator i = components_.begin(); 
        float score = 0.0; 
        for(; i != components_.end(); ++i) { 
            if(i->second->count > 0) { 
                score += CM::score(&(i->second->ss), &hps_); 
            }
        }
        return score; 
           
    }
    

    float post_pred(group_coords_t group_coords, dppos_t dp_pos)
    {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = pdata_[dp_pos]; 
        sswrapper_t * ssw = components_.find(gp)->second; 
        return CM::post_pred(&(ssw->ss), &hps_, val); 
    }
    
    void add_dp(group_coords_t group_coords, dppos_t dp_pos) {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = pdata_[dp_pos]; 
        sswrapper_t * ssw = components_.find(gp)->second; 
        CM::ss_add(&(ssw->ss), &hps_, val); 
        
    }


    void rem_dp(group_coords_t group_coords, dppos_t dp_pos) {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = pdata_[dp_pos]; 
        sswrapper_t * ssw = components_.find(gp)->second; 
        CM::ss_rem(&(ssw->ss), &hps_, val); 
        
    }


private:
    typedef std::map<size_t, sswrapper_t *> components_t; 

    const int NDIM_;
    const typename CM::value_t * pdata_; 
    components_t components_; 
    size_t data_shape_[MAX_AXES]; 

    group_hash_t hash_coords(group_coords_t group_coords) { 
        size_t hash = 0; 
        size_t multiplier = 1; 
        for (int i = 0; i < NDIM_; ++i) { 
            hash += multiplier * (group_coords[i]); 
            multiplier = multiplier * data_shape_[i]; 
        }
        return hash; 

    }
    typename CM::hypers_t hps_; 
}; 

}
