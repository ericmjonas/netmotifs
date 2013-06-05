#ifndef __IRM_COMPONENTMODELS_H__
#define __IRM_COMPONENTMODELS_H__

#include <map>
#include <iostream> 
#include <math.h>
#include <boost/python.hpp>
#include <boost/math/distributions.hpp>

#include "util.h"


namespace bp=boost::python; 


namespace irm { 

typedef size_t group_hash_t; 

inline float betaln(float x, float y) { 
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
    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        if(val) { 
            ss->heads++; 
        } else { 
            ss->tails++; 
        }
    }
    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        if(val) { 
            ss->heads--; 
        } else { 
            ss->tails--; 
        }
        
    }

    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
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

    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data) { 
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

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["heads"] = ss->heads; 
        d["tails"] = ss->tails; 
        return d; 
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
     
    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val,
                       dppos_t dp_pos, RandomAccessIterator data) {
        ss->sum += val; 
        ss->count++; 
    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        ss->sum -= val; 
        ss->count--; 
        
    }

    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
        return val; 
        
    }

    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data) { 
        return ss->sum + hps->offset; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.offset = bp::extract<float>(hps["offset"]); 

        return hp; 

    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["sum"] = ss->sum; 
        d["count"] = ss->count; 
        return d; 
    }

}; 


struct BetaBernoulliNonConj { 
    typedef unsigned char value_t; 
    
    class suffstats_t { 
    public:
        std::vector<uint32_t> datapoint_pos_; 
        float p; 
    }; 

    struct hypers_t { 
        float alpha; 
        float beta; 
    }; 

    static float sample_from_prior(hypers_t * hps, rng_t & rng) {
        float alpha = hps->alpha; 
        float beta = hps->beta; 

        boost::math::beta_distribution<> dist(alpha, beta);
        double p = quantile(dist, uniform_01(rng)); 

        return p; 
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        ss->p = sample_from_prior(hps, rng); 
        ss->datapoint_pos_.reserve(20); 
    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        ss->datapoint_pos_.push_back(dp_pos); 

    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        // FIXME linear search
        auto i = std::find(ss->datapoint_pos_.begin(), 
                           ss->datapoint_pos_.end(), dp_pos); 
        *i = ss->datapoint_pos_.back(); 
        ss->datapoint_pos_.pop_back(); 
    }

    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
        float p = ss->p; 
        if (val) { 
            return logf(p); 
        } else { 
            return logf(1-p); 
        }
        
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float alpha = hps->alpha; 
        float beta = hps->beta; 
        // std::cout << "score_prior " << alpha << " " << beta << std::endl; 
        boost::math::beta_distribution<> dist(alpha, beta);
        if((ss->p > 1.0) || (ss->p < 0.0)) { 
            return -std::numeric_limits<float>::infinity();
        }
        return logf(boost::math::pdf(dist, ss->p)); 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, 
                           RandomAccessIterator data)
    {
        int heads = 0; 
        int tails = 0; 
        if((ss->p > 1.0) || (ss->p < 0.0)) { 
            return -std::numeric_limits<float>::infinity();
        }

        for(auto dpi : ss->datapoint_pos_) { 
            if(data[dpi]) {
                heads++; 
            } else { 
                tails++; 
            }
        }
        boost::math::binomial_distribution<> dist(heads+tails, ss->p); 
        return logf(boost::math::pdf(dist, heads)); 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data) { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, data); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.alpha = bp::extract<float>(hps["alpha"]); 
        hp.beta = bp::extract<float>(hps["beta"]);
        return hp; 

    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["p"] = ss->p; 
        return d; 
    }

}; 




}

#endif
