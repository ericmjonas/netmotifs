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




}

#endif
