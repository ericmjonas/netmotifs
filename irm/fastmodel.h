#ifndef __FASTMODEL_H__
#define __FASTMODEL_H__

#include <map>
#include <iostream> 
#include <math.h>

namespace fastmodel { 

static const size_t MAXDIM = 10; 
typedef size_t group_hash_t; 

float betaln(float x, float y) { 
    return lgamma(x) + lgamma(y) - lgamma(x + y); 

}

struct BetaBernoulli { 
    typedef bool value_t; 
    
    struct suffstats_t { 
        uint32_t heads; 
        uint32_t tails; 
    }; 

    struct hypers_t { 
        float alpha; 
        float beta; 
    }; 
    
    static void ss_init(suffstats_t * ss, hypers_t * hps) { 
        ss->heads = 0; 
        ss->tails = 0; 
    }
     
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val) {
        if(val) { 
            ss->heads++; 
        } else { 
            ss->tails++; 
        }
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val) {
        if(val) { 
            ss->heads--; 
        } else { 
            ss->tails--; 
        }
        
    }

    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) {
        float heads = ss->heads; 
        float tails = ss->tails; 
        float alpha = hps->alpha; 
        float beta = hps->beta; 

        float den = log(alpha + beta + heads + tails); 
        if (val) { 
            return log(heads + alpha) - den; 
        } else { 
            return log(tails + beta) - den; 
        }
        
    }

    static float score(suffstats_t * ss, hypers_t * hps) { 
        float heads = ss->heads; 
        float tails = ss->tails; 
        float alpha = hps->alpha; 
        float beta = hps->beta; 
        
        float logbeta_a_b = betaln(alpha, beta); 
        return betaln(alpha + heads, beta+tails) - logbeta_a_b; 
    }
}; 

template<typename CM>
class  ComponentContainer{ 
    
    struct sswrapper_t { 
        size_t count; 
        typename CM::suffstats_t ss; 
    }; 

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
    
    void create_component(size_t group_coords[]) {
        group_hash_t gp = hash_coords(group_coords); 
        sswrapper_t * ssw = new sswrapper_t; 
        CM::ss_init(&(ssw->ss), &hps_); 
        ssw->count = 0; 
        components_.insert(std::make_pair(gp, ssw)); 
    }

    void delete_component(size_t group_coords[]) {
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
    

    float post_pred(size_t group_coords[], size_t dp_pos)
    {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = pdata_[dp_pos]; 
        sswrapper_t * ssw = components_.find(gp)->second; 
        return CM::post_pred(&(ssw->ss), &hps_, val); 
    }
    
    void add_dp(size_t group_coords[], size_t dp_pos) {
        group_hash_t gp = hash_coords(group_coords); 
        typename CM::value_t val = pdata_[dp_pos]; 
        sswrapper_t * ssw = components_.find(gp)->second; 
        CM::ss_add(&(ssw->ss), &hps_, val); 
        
    }


    void rem_dp(size_t group_coords[], size_t dp_pos) {
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
    size_t data_shape_[MAXDIM]; 

    group_hash_t hash_coords(size_t group_coords[]) {
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

class TestClass {
    typedef int foo_t; 
}; 

template<typename CM>
class Test {
public:
    Test() { 
        std::cout << "Test init" << std::endl; 
    }
    std::string foo() { 
        return "Hello world"; 
    }

}; 



}

#endif
