#ifndef __IRM_FASTMODEL_H__
#define __IRM_FASTMODEL_H__

#include <map>
#include <iostream> 
#include <math.h>
#include <boost/python.hpp>

#include "util.h"


namespace bp=boost::python; 


namespace irm { 

typedef size_t group_hash_t; 

float betaln(float x, float y) { 
    return lgammaf(x) + lgammaf(y) - lgammaf(x + y); 

}

struct BetaBernoulli { 
    typedef unsigned char value_t; 
    
    struct suffstats_t { 
        uint32_t heads; 
        uint32_t tails; 
    }; 

    struct hypers_t { 
        float alpha; 
        float beta; 
    }; 
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
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

        float den = logf(alpha + beta + heads + tails); 
        if (val) { 
            return logf(heads + alpha) - den; 
        } else { 
            return logf(tails + beta) - den; 
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

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.alpha = bp::extract<float>(hps["alpha"]); 
        hp.beta = bp::extract<float>(hps["beta"]);
        return hp; 

    }
}; 

struct AccumModel { 
    typedef float value_t; 
    
    struct suffstats_t { 
        float sum; 
        float count; 
    }; 

    struct hypers_t { 
        float offset; 
    }; 
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                        rng_t & rng) { 
        ss->sum = 0; 
        ss->count = 0; 
    }
     
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val) {
        ss->sum += val; 
        ss->count++; 
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val) {
        ss->sum -= val; 
        ss->count--; 
        
    }

    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) {
        return val; 
        
    }

    static float score(suffstats_t * ss, hypers_t * hps) { 
        return ss->sum + hps->offset; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.offset = bp::extract<float>(hps["offset"]); 

        return hp; 

    }
}; 




}

#endif
