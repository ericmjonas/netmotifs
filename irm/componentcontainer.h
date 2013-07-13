#ifndef __IRM_COMPONENTCONTAINER_H__
#define __IRM_COMPONENTCONTAINER_H__

#include <iostream>
#include <map>
#include <vector>
#include <inttypes.h>
#include <boost/utility.hpp>
#include <boost/python.hpp>

#include "util.h"

#include "componentslice.h"

namespace bp=boost::python; 


namespace irm { 

const static int MAX_GROUPS_PER_AXIS = 256; 

class IComponentContainer {
public:
    virtual size_t dpcount() = 0; 
    virtual float total_score() = 0; 
    virtual void create_component(group_coords_t group_coords, 
                                  rng_t & rng) = 0; 
    virtual void delete_component(group_coords_t group_coords) = 0; 

    virtual float post_pred(group_coords_t group_coords, dppos_t dp_pos) = 0;  
    virtual void add_dp(group_coords_t group_coords, dppos_t dp_pos) = 0; 
    virtual void rem_dp(group_coords_t group_coords, dppos_t dp_pos) = 0; 
    virtual void set_hps(bp::dict & hps) = 0; 
    virtual bp::dict get_hps() = 0; 
    virtual void apply_kernel(std::string name, rng_t & rng, 
                              bp::dict params) = 0; 

    virtual bp::dict get_component(group_coords_t gc) = 0; 
    virtual void set_component(group_coords_t gc, bp::dict val) = 0; 
}; 
   
template<typename CM>
class  ComponentContainer : public IComponentContainer, boost::noncopyable
{ 
    
    struct sswrapper_t { 
        size_t count; 
        typename CM::suffstats_t ss; 
    }; 

    typedef uint64_t group_hash_t; 

public:
    ComponentContainer(const std::string & data, 
                       std::vector<size_t> data_shape) :
        NDIM_(data_shape.size()), 
        data_shape_(data_shape) 
    {
        
        int s = 1; 
        for(int i = 0; i < NDIM_; ++i) { 
            s = s * MAX_GROUPS_PER_AXIS; 
        }

        components_.resize(s); 
        for(int i = 0; i < s; i++) { 
            components_[i] = 0; 
        }

        size_t data_size = 1; 
        for(int i = 0; i < NDIM_; i++) { 
            data_size *= data_shape_[i]; 
        }
        data_.resize(data_size); 
        memcpy(&(data_[0]), data.c_str(), 
               sizeof(typename CM::value_t)*data_size); 
        
    }
    
    ~ComponentContainer() { 
        // for(auto a : components_) { 
        //     delete a.second; 
        // }
        
    }
        
    void create_component(group_coords_t group_coords, 
                          rng_t & rng) { 
        group_hash_t gp = hash_coords(group_coords); 
        // auto i = components_.find(gp); 
        // assert(i == components_.end()); 
        
        sswrapper_t * ssw = new sswrapper_t; 
        CM::ss_sample_new(&(ssw->ss), &hps_, rng); 
        ssw->count = 0; 
        components_[gp] =  ssw; 
    }

    void delete_component(group_coords_t group_coords) {
        group_hash_t gp = hash_coords(group_coords); 
        
        delete components_[gp]; 
        components_[gp] = 0; 
    }
    
    float total_score() {
        typename components_t::iterator i = components_.begin(); 
        float score = 0.0; 
        for(; i != components_.end(); ++i) { 
            if(*i != 0) { 
                if((*i)->count > 0) { 
                    score += CM::score(&((*i)->ss), &hps_, 
                                       data_.begin()); 
                }
            }
        }
        return score; 
           
    }
    

    float post_pred(group_coords_t group_coords, dppos_t dp_pos)
    {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = data_[dp_pos]; 
        sswrapper_t * ssw = components_[gp];
        return CM::post_pred(&(ssw->ss), &hps_, val, 
                             dp_pos, data_.begin()); 
    }
    
    void add_dp(group_coords_t group_coords, dppos_t dp_pos) {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = data_[dp_pos]; 

        sswrapper_t * ssw = components_[gp]; 
        CM::ss_add(&(ssw->ss), &hps_, val, dp_pos, data_.begin()); 
        ssw->count++; 
        
    }


    void rem_dp(group_coords_t group_coords, dppos_t dp_pos) {
        group_hash_t gp = hash_coords(group_coords); 

        typename CM::value_t val = data_[dp_pos]; 
        sswrapper_t * ssw = components_[gp]; 

        CM::ss_rem(&(ssw->ss), &hps_, val, dp_pos, data_.begin()); 
        ssw->count--; 
        
    }

    size_t dpcount() { 
        return data_.size(); 
    }

    void set_hps(bp::dict & hps) { 
        hps_ = CM::bp_dict_to_hps(hps); 

    }

    bp::dict get_hps() { 
        return CM::hps_to_bp_dict(hps_); 

    }

    void apply_kernel(std::string name, rng_t & rng, bp::dict config) { 
        if(name == "slice_sample") { 
            float width = bp::extract<float>(config["width"]); 
            for(auto c : components_) { 
                if(c != 0) { 
                    slice_sample_exec<CM>(rng, width, 
                                          &(c->ss), 
                                          &hps_, data_.begin()); 
                }
            }
        } else { 
            throw std::runtime_error("unknown kernel name"); 
        }


    }

    bp::dict get_component(group_coords_t gc) { 
        group_hash_t gh = hash_coords(gc); 

        sswrapper_t * ssw = components_[gh]; 
        return CM::ss_to_dict(&(ssw->ss)); 
    }

    void set_component(group_coords_t gc, bp::dict val) {
        group_hash_t gh = hash_coords(gc); 

        sswrapper_t * ssw = components_[gh]; 
        CM::ss_from_dict(&(ssw->ss), val); 
        
    }

private:
    typedef std::vector<sswrapper_t *> components_t; 

    const int NDIM_;
    std::vector<size_t> data_shape_; 

    std::vector< typename CM::value_t> data_; 
    components_t components_; 


    group_hash_t hash_coords(group_coords_t group_coords) { 
        size_t hash = 0; 
        size_t multiplier = 1; 
        for (int i = 0; i < NDIM_; ++i) { 
            if(group_coords[i] > (MAX_GROUPS_PER_AXIS-1)) { 
                throw std::runtime_error("Too many groups"); 
            }
            hash += multiplier * (group_coords[i]); 
            multiplier = multiplier * (1<<8);
        }
        return hash; 

    }
    
    typename CM::hypers_t hps_; 

}; 

}

#endif
