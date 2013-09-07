#ifndef __IRM_COMPONENTMODELS_H__
#define __IRM_COMPONENTMODELS_H__

#include <map>
#include <iostream> 
#include <math.h>
#include <boost/python.hpp>
#include <boost/math/distributions.hpp>
#include <unordered_set>
#include <boost/container/flat_set.hpp>

#include "util.h"
#include "fastonebigheader.h"

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
                       RandomAccessIterator data,
                       const std::vector<dppos_t> & dppos) { 
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
    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["alpha"] = hps.alpha; 
        hp["beta"] = hps.beta; 
        
        return hp; 

    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["heads"] = ss->heads; 
        d["tails"] = ss->tails; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->heads = bp::extract<uint32_t>(v["heads"]); 
        ss->tails = bp::extract<uint32_t>(v["tails"]); 
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
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos) { 
        return ss->sum + hps->offset; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.offset = bp::extract<float>(hps["offset"]); 

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["offset"] = hps.offset; 
        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["sum"] = ss->sum; 
        d["count"] = ss->count; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        throw std::runtime_error("Not Implemented"); 
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

        boost::math::beta_distribution<> dist(alpha, beta);
        if((ss->p >= 1.0) || (ss->p < 0.0)) { 
            return -std::numeric_limits<float>::infinity();
        }

        return logf(boost::math::pdf(dist, ss->p)); 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, 
                                  RandomAccessIterator data, 
                                  const std::vector<dppos_t> & dppos)
    {
        // int heads = 0; 
        // int tails = 0; 
        // if((ss->p > 1.0) || (ss->p < 0.0)) { 
        //     return -std::numeric_limits<float>::infinity();
        // }

        // for(auto dpi : ss->datapoint_pos_) { 
        //     if(data[dpi]) {
        //         heads++; 
        //     } else { 
        //         tails++; 
        //     }
        // }
        // boost::math::binomial_distribution<> dist(heads+tails, ss->p); 
        // return logf(boost::math::pdf(dist, heads)); 
        float score = 0.0; 
        for(auto dpi : dppos) { 
            if(data[dpi]) {
                score += logf(ss->p); 
            } else { 
                score += logf(1 - ss->p); 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos) { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.alpha = bp::extract<float>(hps["alpha"]); 
        hp.beta = bp::extract<float>(hps["beta"]);
        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["alpha"] = hps.alpha; 
        hp["beta"] = hps.beta; 
        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["p"] = ss->p; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->p = bp::extract<float>(v["p"]); 

    }

}; 


struct LogisticDistance { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    class suffstats_t { 
    public:
        float mu; 
        float lambda; 
    }; 

    class hypers_t {
    public:
        // NOTE : BE CAREFUL there are multiple parameterizations of the 
        // expoentnail distribution here. np expects 1/lamb, boost expects lamb
        float mu_hp; // mu_hp : with our exponential prior, this is the mean. 
        float lambda_hp; // for our lambda prior, this is the mean
        float p_min; 
        float p_max; 
        inline hypers_t() : 
            mu_hp(1.0), 
            lambda_hp(1.0), 
            p_min(0.1), 
            p_max(0.9)
        { 


        }
    }; 

    static float rev_logistic_scaled(float x, float mu, float lambda, 
                                     float pmin, float pmax) { 
        // reversed logistic function 
        float ratio = (x-mu)/lambda; 
        float p_unscaled = 1.0/(1.0 + MYEXP(ratio)); 
        return p_unscaled * (pmax-pmin) + pmin;         
    }

    static std::pair<float, float> sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float lambda_hp = hps->lambda_hp; 
        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 
            boost::math::exponential_distribution<> lamb_dist(1.0/lambda_hp);
            float lamb = quantile(lamb_dist, r2); 
            return std::make_pair(mu, lamb); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp << " lambda_hp=" << lambda_hp
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->lambda = params.second; 
    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
        float mu = ss->mu; 
        float lambda = ss->lambda; 
        
        float p = rev_logistic_scaled(val.distance, mu, lambda, 
                                      hps->p_min, hps->p_max); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float lambda = ss->lambda; 
        float mu_hp = hps->mu_hp; 
        float lambda_hp = hps->lambda_hp; 
        float score = 0.0; 
        score += log_exp_dist(mu, 1./mu_hp); 
        score += log_exp_dist(lambda, 1./lambda_hp); 
        return score; 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            float p = rev_logistic_scaled(data[dpi].distance, ss->mu, 
                                          ss->lambda, hps->p_min, 
                                          hps->p_max); 
            //p = p * p_range + hps->p_min;  // already calculated
            float lscore; 
            if(data[dpi].link) {
                lscore = MYLOG(p); 
            } else { 
                lscore = MYLOG(1 - p); 
            }
            score += lscore; 

        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.lambda_hp = bp::extract<float>(hps["lambda_hp"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);
        hp.p_max = bp::extract<float>(hps["p_max"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["lambda_hp"] = hps.lambda_hp; 
        hp["p_min"] = hps.p_min; 
        hp["p_max"] = hps.p_max; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["lambda"] = ss->lambda; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->lambda = bp::extract<float>(v["lambda"]); 
            
    }

}; 


struct SigmoidDistance { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    class suffstats_t { 
    public:
        float mu; 
        float lambda; 
    }; 

    class hypers_t {
    public:
        float mu_hp; 
        float lambda_hp; 
        float p_min; 
        float p_max; 
        inline hypers_t() : 
            mu_hp(1.0), 
            lambda_hp(1.0), 
            p_min(0.1), 
            p_max(0.9)
        { 


        }
    }; 

    static float sigmoid_scaled(float x, float mu, float lambda, 
                                float pmin, float pmax) { 
        float p_unscaled = (x-mu)/(lambda + fabsf((x-mu))) * 0.5 + 0.5; 
        return p_unscaled * (pmax-pmin) + pmin;         
    }

    static std::pair<float, float> sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float lambda_hp = hps->lambda_hp; 
        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1./mu_hp);
            float mu = quantile(mu_dist, r1); 
            boost::math::exponential_distribution<> lamb_dist(1./lambda_hp);
            float lamb = quantile(lamb_dist, r2); 
            return std::make_pair(mu, lamb); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp << " lambda_hp=" << lambda_hp << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->lambda = params.second; 

    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
    }


    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
        float mu = ss->mu; 
        float lambda = ss->lambda; 
        
        float p = sigmoid_scaled(val.distance, mu, lambda, 
                                      hps->p_min, hps->p_max); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float lambda = ss->lambda; 
        float mu_hp = hps->mu_hp; 
        float lambda_hp = hps->lambda_hp; 
        float score = 0.0; 
        score += log_exp_dist(mu, 1./mu_hp); 
        score += log_exp_dist(lambda, 1./lambda_hp); 
        return score; 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 
        for(auto dpi : dppos) { 
            float p = sigmoid_scaled(data[dpi].distance, ss->mu, 
                                     ss->lambda, hps->p_min, 
                                     hps->p_max); 
            float lscore; 
            if(data[dpi].link) {
                lscore = logf(p); 
            } else { 
                lscore = logf(1 - p); 
            }
            score += lscore; 

        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.lambda_hp = bp::extract<float>(hps["lambda_hp"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);
        hp.p_max = bp::extract<float>(hps["p_max"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["lambda_hp"] = hps.lambda_hp; 
        hp["p_min"] = hps.p_min; 
        hp["p_max"] = hps.p_max; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["lambda"] = ss->lambda; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->lambda = bp::extract<float>(v["lambda"]); 
            
    }

}; 

struct LinearDistance { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    class suffstats_t { 
    public:
        //std::unordered_set<uint32_t> datapoint_pos_; 
        float mu; 
        float p; 
    }; 

    class hypers_t {
    public:
        float mu_hp; 
        float p_alpha; 
        float p_beta; 
        float p_min; 
        inline hypers_t() : 
            mu_hp(1.0), 
            p_alpha(1.0), 
            p_beta(1.0), 
            p_min(0.01)
        { 


        }
    }; 

    static float linear_prob(float x, float mu, float p, float p_min) { 
        if (x > mu) { 
            return p_min; 
        } 
        return -p / mu * x + p; 
    }

    static std::pair<float, float> 
    sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float p_alpha = hps->p_alpha; 
        float p_beta = hps->p_beta;

        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 

            
            boost::math::beta_distribution<> beta_dist(p_alpha, p_beta);
            float p = quantile(beta_dist, r2); 
            
            return std::make_pair(mu, p); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp 
                      << " p_alpha=" << p_alpha << " p_beta=" << p_beta
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->p = params.second; 
    
    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {

        float p = linear_prob(val.distance, ss->mu, ss->p, 
                              hps->p_min); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float p = ss->p; 
        if((p >= 1.0) || (p < 0.0) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_beta_dist(p, hps->p_alpha, hps->p_beta); 
        return score; 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            float p = linear_prob(data[dpi].distance, ss->mu, 
                                  ss->p, hps->p_min); 

            float lscore; 
            if(data[dpi].link) {
                lscore = logf(p); 
            } else { 
                lscore = logf(1 - p); 
            }
            score += lscore; 

        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.p_alpha = bp::extract<float>(hps["p_alpha"]);
        hp.p_beta = bp::extract<float>(hps["p_beta"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["p_alpha"] = hps.p_alpha; 
        hp["p_beta"] = hps.p_beta; 
        hp["p_min"] = hps.p_min; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["p"] = ss->p; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->p = bp::extract<float>(v["p"]); 
            
    }

}; 


struct GammaPoisson { 
    typedef uint32_t value_t; 
    
    struct suffstats_t { 
        uint32_t n; 
        uint32_t sum; 
        float log_prod; 
    }; 

    struct hypers_t { 
        float alpha; 
        float beta; 
    }; 
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        ss->n = 0; 
        ss->sum = 0; 
        ss->log_prod = 0.0; 
    }
    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        ss->n++; 
        ss->sum += val; 
        ss->log_prod += log_factorial(val); 
    }
    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        ss->n--; 
        ss->sum -= val; 
        ss->log_prod -= log_factorial(val);
        
    }

    static std::pair<float, float> intermediates(hypers_t * hps, suffstats_t * ss)
    {
        float alpha_n = hps->alpha + ss->sum; 
        float beta_n = 1.0 / (ss->n + 1. / hps->beta); 
        return std::make_pair(alpha_n, beta_n); 
    }

    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {
        
        auto im = intermediates(hps, ss); 
        float alpha_z = im.first; 
        float beta_z = im.second; 
        
        return lgamma(alpha_z + val) - lgamma(alpha_z) - alpha_z * log(beta_z) + (alpha_z + val) * log(1.0 / (1.0 + 1./beta_z)) - log_factorial(val);

    }

    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data,
                       const std::vector<dppos_t> & dppos) { 
        auto im = intermediates(hps, ss); 
        float alpha_z = im.first; 
        float beta_z = im.second; 
        return lgamma(alpha_z) - lgamma(hps->alpha) + alpha_z*log(beta_z) - hps->alpha * log(hps->beta) - ss->log_prod; 

    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.alpha = bp::extract<float>(hps["alpha"]); 
        hp.beta = bp::extract<float>(hps["beta"]);
        return hp; 

    }
    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["alpha"] = hps.alpha; 
        hp["beta"] = hps.beta; 
        
        return hp; 

    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["n"] = ss->n; 
        d["sum"] = ss->sum; 
        d["log_prod"] = ss->log_prod; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->n = bp::extract<uint32_t>(v["n"]); 
        ss->sum = bp::extract<uint32_t>(v["sum"]); 
        ss->log_prod = bp::extract<float>(v["log_prod"]); 

    }

}; 


struct NormalDistanceFixedWidth { 
    /* 
       Move a normal-like bump around varying 
       both the height and the offset position
       
       Inference is done on the height and the pos but not the width
       

     */ 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    class suffstats_t { 
    public:
        //std::unordered_set<uint32_t> datapoint_pos_; 
        float mu; 
        float p; 
    }; 

    class hypers_t {
    public:
        float mu_hp; // exp pos
        float width; 
        float p_alpha; 
        float p_beta; 
        float p_min; 
        inline hypers_t() : 
            mu_hp(1.0), 
            width(0.1), 
            p_alpha(1.0), 
            p_beta(1.0), 
            p_min(0.01)
        { 


        }
    }; 

    static float norm_prob(float x, float mu, 
                           float p, float p_min, float width) { 
        float my_p = MYEXP(-0.5*(x - mu)*(x-mu)/(width*width)) * p + p_min; 
        float max_p = 1.0-p_min; 
        if (my_p > max_p)
            return max_p; 
        return my_p; 

    }

    static std::pair<float, float> 
    sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float p_alpha = hps->p_alpha; 
        float p_beta = hps->p_beta;

        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 

            
            boost::math::beta_distribution<> beta_dist(p_alpha, p_beta);
            float p = quantile(beta_dist, r2); 
            
            return std::make_pair(mu, p); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp 
                      << " p_alpha=" << p_alpha << " p_beta=" << p_beta
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->p = params.second; 
    
    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {

        float p = norm_prob(val.distance, ss->mu, ss->p, 
                            hps->p_min, hps->width); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float p = ss->p; 
        if((p >= 1.0) || (p < 0.0) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_beta_dist(p, hps->p_alpha, hps->p_beta); 
        return score; 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            float p = norm_prob(data[dpi].distance, ss->mu, 
                                  ss->p, hps->p_min, hps->width); 

            float lscore; 
            if(data[dpi].link) {
                lscore = logf(p); 
            } else { 
                lscore = logf(1 - p); 
            }
            score += lscore; 

        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.p_alpha = bp::extract<float>(hps["p_alpha"]);
        hp.p_beta = bp::extract<float>(hps["p_beta"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);
        hp.width = bp::extract<float>(hps["width"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["p_alpha"] = hps.p_alpha; 
        hp["p_beta"] = hps.p_beta; 
        hp["p_min"] = hps.p_min; 
        hp["width"] = hps.width; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["p"] = ss->p; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->p = bp::extract<float>(v["p"]); 
            
    }

}; 

struct SquareDistanceBump { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    class suffstats_t { 
    public:
        //std::unordered_set<uint32_t> datapoint_pos_; 
        float mu; 
        float p; 
    }; 

    class hypers_t {
    public:
        float mu_hp; 
        float p_alpha; 
        float p_beta; 
        float p_min; 
        float param_weight; 
        float param_max_distance; 
        inline hypers_t() : 
            mu_hp(1.0), 
            p_alpha(1.0), 
            p_beta(1.0), 
            p_min(0.001), 
            param_weight(0.5),
            param_max_distance(1e100)
        { 


        }
    }; 

    static float square_prob(float x, float mu, float p, float p_min) { 
        if (x > mu) { 
            return p_min; 
        } 
        return p; 
    }

    static std::pair<float, float> 
    sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float p_alpha = hps->p_alpha; 
        float p_beta = hps->p_beta;
        float p_min = hps->p_min; 
        float param_weight = hps->param_weight; 
        float param_max_distance = hps->param_max_distance; 
        
        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        float r3 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 
            if (r2 < param_weight) { 
                mu = param_max_distance; 
            }
            
            boost::math::beta_distribution<> beta_dist(p_alpha, p_beta);
            float p = quantile(beta_dist, r3); 
            
            return std::make_pair(mu, p); 

        } catch (...){
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->p = params.second; 
    
    }

    template<typename RandomAccessIterator>
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    template<typename RandomAccessIterator>
    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos, RandomAccessIterator data) {
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    template<typename RandomAccessIterator>
    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val, 
                           dppos_t dp_pos, RandomAccessIterator data) {

        float p = square_prob(val.distance, ss->mu, ss->p, 
                              hps->p_min); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float p = ss->p; 

        if((p >= 1.0) || (p < 0.0) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = log(1.0 - hps->param_weight)  + log_exp_dist(mu, 1./hps->mu_hp); 
        if(mu == hps->param_max_distance) { 
            score  = log_sum_exp(score, log(hps->param_weight)  + log(hps->param_weight)); 
        }
        

        float p_score = log_beta_dist(p, hps->p_alpha, hps->p_beta); 
        return score + p_score; 
    }
    
    template<typename RandomAccessIterator> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            float p = square_prob(data[dpi].distance, ss->mu, 
                                  ss->p, hps->p_min); 

            float lscore; 
            if(data[dpi].link) {
                lscore = logf(p); 
            } else { 
                lscore = logf(1 - p); 
            }
            score += lscore; 

        }
        return score; 
    }
    
    template<typename RandomAccessIterator>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.p_alpha = bp::extract<float>(hps["p_alpha"]);
        hp.p_beta = bp::extract<float>(hps["p_beta"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);
        hp.param_weight = bp::extract<float>(hps["param_weight"]);
        hp.param_max_distance = bp::extract<float>(hps["param_max_distance"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["p_alpha"] = hps.p_alpha; 
        hp["p_beta"] = hps.p_beta; 
        hp["p_min"] = hps.p_min; 
        hp["param_weight"] = hps.param_weight; 
        hp["param_max_distance"] = hps.param_max_distance; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["p"] = ss->p; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->p = bp::extract<float>(v["p"]); 
            
    }

}; 


}

#endif
