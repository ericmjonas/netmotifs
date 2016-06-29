#ifndef __IRM_COMPONENTMODELS_H__
#define __IRM_COMPONENTMODELS_H__

#include <map>
#include <iostream> 
#include <math.h>
#include <limits>
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
    static const bool is_addrem_mutating = true; 

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
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        if(val) { 
            ss->heads++; 
        } else { 
            ss->tails++; 
        }
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
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

        float den = MYLOG(alpha + beta + heads + tails); 
        if (val) { 
            return MYLOG(heads + alpha) - den; 
        } else { 
            return MYLOG(tails + beta) - den; 
        }
        
    }

    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data,
                       RandomAccessIterator2 observed,
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
    static const bool is_addrem_mutating = true; 
    
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
     
    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val,
                       dppos_t dp_pos) { 
        ss->sum += val; 
        ss->count++; 
    }


    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        ss->sum -= val; 
        ss->count--; 
        
    }

    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        return val; 
        
    }

    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
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
    static const bool is_addrem_mutating = false; 
    
    class suffstats_t { 
    public:
        float p; 
    }; 

    struct hypers_t { 
        float alpha; 
        float beta; 
    }; 

    static float sample_from_prior(hypers_t * hps, rng_t & rng) {
        try { 
            float alpha = hps->alpha; 
            float beta = hps->beta;
            
            if(alpha <= 0) { 
                throw std::runtime_error("Cannot sample with alpha <= 0"); 
            }
            
            if(beta <= 0) { 
                throw std::runtime_error("Cannot sample with alpha <= 0"); 
            }
            
            float p = beta_sample(alpha, beta, rng);
            if ((p <= 0) || (p >= 1.0)) {
                std::cerr << "p sampled outside range p=" << p << std::endl 
            }
            
            //boost::math::beta_distribution<> dist(alpha, beta);
            //double p = quantile(dist, uniform_01(rng)); 
            
            return p;
        } catch (std::exception & e) {
            std::cerr << "Error in sample_from_prior " <<  e.what() << std::endl;
            throw; 

        }
        
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        ss->p = sample_from_prior(hps, rng); 

    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        // FIXME linear search
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        float p = ss->p; 
        if (val) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
        
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float alpha = hps->alpha; 
        float beta = hps->beta; 
        try { 
            if((alpha <= 0) || (beta <= 0)) { 
                return -std::numeric_limits<float>::infinity();
            }
            
            boost::math::beta_distribution<> dist(alpha, beta);
            if((ss->p >= 1.0) || (ss->p <= 0.0)) { 
                return -std::numeric_limits<float>::infinity();
            }
        
            return MYLOG(boost::math::pdf(dist, ss->p));
        } catch (std::exception & e) {
            std::cerr << "ss->p= " << ss->p << " alpha=" << alpha << " beta=" << beta << std::endl; 
            std::cerr << "Error in score_prior " << e.what() << std::endl;
            throw; 
        }
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
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
            if(observed[dpi]) { 
                if(data[dpi]) {
                    score += MYLOG(ss->p); 
                } else { 
                    score += MYLOG(1 - ss->p); 
                }
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos) { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, data, observed, dppos); 
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

    static const bool is_addrem_mutating = false; 
    
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

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }


    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }



    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

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
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if(observed[dpi]) { 
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
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, observed, 
                                                  dppos); 
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

struct LogisticDistanceFixedLambda { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 
    
    static const bool is_addrem_mutating = false; 

    class suffstats_t { 
    public:
        float mu; 
        float p_scale; 
    }; 

    class hypers_t {
    public:
        // NOTE : BE CAREFUL there are multiple parameterizations of the 
        // expoentnail distribution here. np expects 1/lamb, boost expects lamb
        float mu_hp; // mu_hp : with our exponential prior, this is the mean. 
        float p_scale_alpha_hp; 
        float p_scale_beta_hp; 
        float p_min; 
        float lambda; 
        inline hypers_t() : 
            mu_hp(1.0), 
            lambda(1.0), 
            p_min(0.1), 
            p_scale_alpha_hp(1.0), 
            p_scale_beta_hp(1.0)
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
        float alpha = hps->p_scale_alpha_hp; 
        float beta = hps->p_scale_beta_hp; 

        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 

            double p = beta_sample(alpha, beta, rng); 
            if (p < 0.0001) 
                p = 0.0001; 
            if (p > 0.9999)
                p = 0.9999; 
                    
            return std::make_pair(mu, p); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp 
                      << "p_scale_alpha_hp=" << alpha
                      << "p_scale_beta_hp=" << beta
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->p_scale = params.second; 
    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        float mu = ss->mu; 
        float p_scale = ss->p_scale; 
        
        float p = rev_logistic_scaled(val.distance, mu, hps->lambda, 
                                      hps->p_min, p_scale); 

        if (val.link) { 
            return MYLOG(p); 
        } else { 
            return MYLOG(1-p); 
        }
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float p_scale = ss->p_scale; 
        float mu_hp = hps->mu_hp; 
        float alpha = hps->p_scale_alpha_hp; 
        float beta = hps->p_scale_beta_hp; 
        float score = 0.0; 
        score += log_exp_dist(mu, 1./mu_hp); 

        boost::math::beta_distribution<> dist(alpha, beta);
        if((p_scale > 0.99999) || (p_scale < 0.00001)) { // safety
            return -std::numeric_limits<float>::infinity();
        }

        score += MYLOG(boost::math::pdf(dist, p_scale)); 

        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if(observed[dpi]) { 
                float p = rev_logistic_scaled(data[dpi].distance, ss->mu, 
                                              hps->lambda, hps->p_min, 
                                              ss->p_scale); 
                //p = p * p_range + hps->p_min;  // already calculated
                float lscore; 
                if(data[dpi].link) {
                    lscore = MYLOG(p); 
                } else { 
                    lscore = MYLOG(1 - p); 
                }
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, observed, 
                                                  dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.lambda = bp::extract<float>(hps["lambda"]);
        hp.p_min = bp::extract<float>(hps["p_min"]);
        hp.p_scale_alpha_hp = bp::extract<float>(hps["p_scale_alpha_hp"]);
        hp.p_scale_beta_hp = bp::extract<float>(hps["p_scale_beta_hp"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["lambda"] = hps.lambda; 
        hp["p_min"] = hps.p_min; 
        hp["p_scale_alpha_hp"] = hps.p_scale_alpha_hp; 
        hp["p_scale_beta_hp"] = hps.p_scale_beta_hp; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["p_scale"] = ss->p_scale; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->p_scale = bp::extract<float>(v["p_scale"]); 
            
    }

}; 


struct SigmoidDistance { 
    class value_t {
    public:
        char link; 
        float distance; 
    } __attribute__((packed)); 

    static const bool is_addrem_mutating = false; 
    
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

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
    }



    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        
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
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 
        for(auto dpi : dppos) { 
            if (observed[dpi]) { 
                float p = sigmoid_scaled(data[dpi].distance, ss->mu, 
                                         ss->lambda, hps->p_min, 
                                         hps->p_max); 
                float lscore; 
                if(data[dpi].link) {
                    lscore = MYLOG(p); 
                } else { 
                    lscore = MYLOG(1 - p); 
                }
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 
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
    
    static const bool is_addrem_mutating = false; 


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
            float p = beta_sample(p_alpha, p_beta, rng); 
            
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

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

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
        if((p > 0.99999999) || (p <= 0.0000001) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_beta_dist(p, hps->p_alpha, hps->p_beta); 
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if(observed[dpi]) { 
                float p = linear_prob(data[dpi].distance, ss->mu, 
                                      ss->p, hps->p_min); 

                float lscore; 
                if(data[dpi].link) {
                    lscore = MYLOG(p); 
                } else { 
                    lscore = MYLOG(1 - p); 
                }
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 

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
    static const bool is_addrem_mutating = true; 

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


    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        ss->n++; 
        ss->sum += val; 
        ss->log_prod += log_factorial(val); 
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
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


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        
        auto im = intermediates(hps, ss); 
        float alpha_z = im.first; 
        float beta_z = im.second; 
        
        return lgammaf(alpha_z + val) - lgammaf(alpha_z) - alpha_z * MYLOG(beta_z) + (alpha_z + val) * MYLOG(1.0 / (1.0 + 1./beta_z)) - log_factorial(val);

    }

    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data,
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos) { 
        auto im = intermediates(hps, ss); 
        float alpha_z = im.first; 
        float beta_z = im.second; 
        return lgammaf(alpha_z) - lgammaf(hps->alpha) + alpha_z*log(beta_z) - hps->alpha * log(hps->beta) - ss->log_prod; 

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

    static const bool is_addrem_mutating = false; 
    
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

            float p = beta_sample(p_alpha, p_beta, rng); 
            if (p < 0.0001) 
                p = 0.0001; 
            if (p > 0.9999)
                p = 0.9999; 
            
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

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

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
        if((p >= 1.0) || (p <= 0.0) || mu <= 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_beta_dist(p, hps->p_alpha, hps->p_beta); 
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if (observed[dpi]) { 
                float p = norm_prob(data[dpi].distance, ss->mu, 
                                    ss->p, hps->p_min, hps->width); 

                float lscore; 
                if(data[dpi].link) {
                    lscore = MYLOG(p); 
                } else { 
                    lscore = MYLOG(1 - p); 
                }
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 
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
    
    static const bool is_addrem_mutating = false; 

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
            float p = beta_sample(p_alpha, p_beta, rng); 
            
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

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }



    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

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
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if(observed[dpi]) { 
                float p = square_prob(data[dpi].distance, ss->mu, 
                                      ss->p, hps->p_min); 

                float lscore; 
                if(data[dpi].link) {
                    lscore = MYLOG(p); 
                } else { 
                    lscore = MYLOG(1 - p); 
                }
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 
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


struct ExponentialDistancePoisson { 
    /*
      Link function is exp(dist, 1./mu)*rate

      prior on rate is exp with really hith mean
      prior mu is again, long exp

     */
    class value_t {
    public:
        int32_t link; 
        float distance; 
    } __attribute__((packed)); 
    
    static const bool is_addrem_mutating = false; 

    class suffstats_t { 
    public:
        //std::unordered_set<uint32_t> datapoint_pos_; 
        float mu; 
        float rate_scale; 
    }; 

    class hypers_t {
    public:
        float mu_hp; 
        float rate_scale_hp; 
        inline hypers_t() : 
            mu_hp(1.0), 
            rate_scale_hp(1.0) 
        { 


        }
    }; 

    static float exp_rate(float x, float mu, float rate_scale) { 
        float r = expf(-x / mu); 
        float r_scaled = r * rate_scale; 
        const float RATE_MIN = 0.000001; 
        if(r_scaled < RATE_MIN) 
            return RATE_MIN; 
        return r_scaled; 
    }

    static std::pair<float, float> 
    sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float rate_scale_hp = hps->rate_scale_hp;

        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 

            
            boost::math::exponential_distribution<> rate_scale_dist(1.0/rate_scale_hp); 
            float rate_scale = quantile(rate_scale_dist, r2); 
            
            return std::make_pair(mu, rate_scale); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp 
                      << " rate_hp=" << rate_scale_hp 
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->rate_scale = params.second; 
    
    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos){ 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

        float rate = exp_rate(val.distance, ss->mu, ss->rate_scale); 

        return log_poisson_dist(val.link, rate); 
    }
    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float rate_scale = ss->rate_scale; 
        if( (rate_scale < 0.0) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_exp_dist(rate_scale, 1.0/hps->rate_scale_hp); 
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if (observed[dpi]) { 
                
                float rate = exp_rate(data[dpi].distance, ss->mu, 
                                      ss->rate_scale); 
                float lscore = log_poisson_dist(data[dpi].link, rate); 
                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.rate_scale_hp = bp::extract<float>(hps["rate_scale_hp"]);

        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["rate_scale_hp"] = hps.rate_scale_hp; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["rate_scale"] = ss->rate_scale; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->rate_scale = bp::extract<float>(v["rate_scale"]); 
            
    }

}; 


struct LogisticDistancePoisson { 
    class value_t {
    public:
        int32_t link; 
        float distance; 
    } __attribute__((packed)); 
    
    static const bool is_addrem_mutating = false; 

    class suffstats_t { 
    public:
        float mu; 
        float rate_scale; 
    }; 

    class hypers_t {
    public:
        // NOTE : BE CAREFUL there are multiple parameterizations of the 
        // expoentnail distribution here. np expects 1/lamb, boost expects lamb
        float mu_hp; // mu_hp : with our exponential prior, this is the mean. 
        float rate_scale_hp; 
        float rate_min; 
        float lambda; 
        inline hypers_t() : 
            mu_hp(1.0), 
            lambda(1.0), 
            rate_min(0.1), 
            rate_scale_hp(1.0)
        { 


        }
    }; 

    static float rev_logistic_scaled(float x, float mu, float lambda, 
                                     float pmin, float pmax) { 
        // reversed logistic function 
        float ratio = (x-mu)/lambda; 
        float rate_unscaled = 1.0/(1.0 + MYEXP(ratio)); 
        return rate_unscaled * (pmax-pmin) + pmin;         
    }

    static std::pair<float, float> sample_from_prior(hypers_t * hps, rng_t & rng) {
        float mu_hp = hps->mu_hp; 
        float rate_scale_hp = hps->rate_scale_hp; 

        float r1 = uniform_01(rng); 
        float r2 = uniform_01(rng); 
        
        try { 
            boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
            float mu = quantile(mu_dist, r1); 

            boost::math::exponential_distribution<> pscale_dist(1.0/rate_scale_hp);
            float rate_scale = quantile(pscale_dist, r2); 

            if (rate_scale < 0.0001) 
                rate_scale = 0.0001; 
                    
            return std::make_pair(mu, rate_scale); 

        } catch (...){
            
            std::cout << "mu_hp=" << mu_hp 
                      << "rate_scale_hp=" << rate_scale_hp
                      << std::endl; 
            std::cout << "r1=" << r1 << " r2=" << r2 << std::endl; 
            throw std::runtime_error("Sample from prior error"); 

        }
    }
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        std::pair<float, float> params = sample_from_prior(hps, rng); 
        ss->mu = params.first; 
        ss->rate_scale = params.second; 
    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.insert(dp_pos); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        //ss->datapoint_pos_.erase(dp_pos); 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 

        float mu = ss->mu; 
        float rate_scale = ss->rate_scale; 
        
        float rate = rev_logistic_scaled(val.distance, mu, hps->lambda, 
                                         hps->rate_min, rate_scale); 
        
        return log_poisson_dist(val.link, rate); 
    }

    
    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        float mu = ss->mu; 
        float rate_scale = ss->rate_scale; 
        if( (rate_scale < 0.0) || mu < 0.0) { 
            return -std::numeric_limits<float>::infinity();
        }

        float score = 0.0; 
        score += log_exp_dist(mu, 1./hps->mu_hp); 
        score += log_exp_dist(rate_scale, 1.0/hps->rate_scale_hp); 
        return score; 

    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 

        for(auto dpi : dppos) { 
            if (observed[dpi]) { 
                float rate = rev_logistic_scaled(data[dpi].distance, ss->mu, 
                                                 hps->lambda, hps->rate_min, 
                                                 ss->rate_scale); 
                //p = p * p_range + hps->p_min;  // already calculated
                float lscore = log_poisson_dist(data[dpi].link, rate); 

                score += lscore; 
            }
        }
        return score; 
    }
    
    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data, 
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos)
    { 
        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, 
                                                  observed, dppos); 
        return prior_score + likelihood_score; 
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu_hp = bp::extract<float>(hps["mu_hp"]); 
        hp.lambda = bp::extract<float>(hps["lambda"]);
        hp.rate_min = bp::extract<float>(hps["rate_min"]);
        hp.rate_scale_hp = bp::extract<float>(hps["rate_scale_hp"]);
        return hp; 

    }

    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu_hp"] = hps.mu_hp; 
        hp["lambda"] = hps.lambda; 
        hp["rate_min"] = hps.rate_min; 
        hp["rate_scale_hp"] = hps.rate_scale_hp; 

        return hp; 
    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = ss->mu; 
        d["rate_scale"] = ss->rate_scale; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = bp::extract<float>(v["mu"]); 
        ss->rate_scale = bp::extract<float>(v["rate_scale"]); 
            
    }

}; 

struct NormalInverseChiSq { 
    typedef float value_t; 

    static const bool is_addrem_mutating = true; 
    
    struct suffstats_t { 
        uint32_t count; 
        float mean; 
        float var; 
    }; 

    struct hypers_t { 
        float mu; 
        float kappa; 
        float sigmasq; 
        float nu; 
    }; 
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 
        ss->count = 0; 
        ss->mean = 0.0; 
        ss->var = 0.0; 

    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        ss->count++; 
        float delta = val - ss->mean; 
        ss->mean += delta/ss->count; 
        ss->var += delta * (val - ss->mean); 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
        float total = ss->mean * ss->count; 
        float delta = val - ss->mean; 
        ss->count--; 
        if (ss->count == 0) { 
            ss->mean = 0; 
        } else { 
            ss->mean = (total - val) / ss->count; 
        }

        if (ss->count <= 1) {
            ss->var = 0; 
        } else {
            ss->var -= delta * (val - ss->mean); 
        }

        
    }

    static std::tuple<float, float, float, float> 
    intermediates(hypers_t * hps, 
                  suffstats_t * ss) { 
        float total = ss->mean * ss->count; 
        float mu_1 = hps->mu - ss->mean; 
        float kappa_n = hps->kappa + ss->count; 
        float mu_n = (hps->kappa * hps->mu + total) / kappa_n; 
        float nu_n = hps->nu + ss->count; 
        float sigmasq_n = 1. / nu_n * (
                                       hps->nu * hps->sigmasq 
                                       + ss->var
                                       + (ss->count * hps->kappa * mu_1 * mu_1)/kappa_n); 
        return std::make_tuple(mu_n, kappa_n, sigmasq_n, nu_n); 

    }

    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) { 
        float mu_n, kappa_n, sigmasq_n, nu_n; 
        std::tie(mu_n, kappa_n, sigmasq_n, nu_n) = intermediates(hps, ss); 
        return log_t_pdf(
                         val,
                         nu_n,
                         mu_n,
                         ((1 + kappa_n) * sigmasq_n) / kappa_n); 

        
    }

    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data,
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos) { 
        float mu_n, kappa_n, sigmasq_n, nu_n; 
        std::tie(mu_n, kappa_n, sigmasq_n, nu_n) = intermediates(hps, ss); 
        return lgammaf(nu_n/2.0f) - lgammaf(hps->nu/2.0) + 
            0.5f * MYLOG(hps->kappa / kappa_n) + 
            (0.5f * hps->nu) * MYLOG(hps->nu * hps->sigmasq) - 
            (0.5f * nu_n) * MYLOG(nu_n * sigmasq_n) -
            ss->count/2.0 * 1.1447298858493991; 
     
    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.mu = bp::extract<float>(hps["mu"]); 
        hp.kappa = bp::extract<float>(hps["kappa"]);
        hp.sigmasq = bp::extract<float>(hps["sigmasq"]);
        hp.nu = bp::extract<float>(hps["nu"]);
        return hp; 

    }
    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["mu"] = hps.mu; 
        hp["kappa"] = hps.kappa; 
        hp["sigmasq"] = hps.sigmasq; 
        hp["nu"] = hps.nu; 
        
        return hp; 

    }

    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["count"] = ss->count; 
        d["mean"] = ss->mean; 
        d["var"] = ss->var; 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->count = bp::extract<uint32_t>(v["count"]); 
        ss->mean = bp::extract<float>(v["mean"]); 
        ss->var = bp::extract<float>(v["var"]); 
    }

}; 

struct MixtureModelDistribution { 
    /* 
       nonconjugate type that includes 
       
     */ 
    static const size_t MAX_DP = 1024; 
    class value_t { 
    public: // whatever there are going to be alignment issues in memory
        int size_; 
        float data_[MAX_DP]; 

        inline size_t size() const { 
            return size_; 
        }

         inline float  operator[](size_t pos) const { 
            return data_[pos]; 

        }
    }; 
    static const bool is_addrem_mutating = false; 

    static constexpr float CHI_VAL = 1.0 ; 
    static constexpr   float EPSILON = 0.0001; 
    struct suffstats_t { 
        // parameters
        std::vector<float> mu; 
        std::vector<float> var; 
        std::vector<float> pi; 
        
    }; 

    struct hypers_t { 
        int comp_k; 
        float dir_alpha; 
        float var_scale; 
    }; 
    
    static void ss_sample_new(suffstats_t * ss, hypers_t * hps, 
                              rng_t & rng) { 

        auto s =  sample_from_prior(hps, rng); 
        ss->mu = s.mu; 
        ss->var = s.var; 
        ss->pi = s.pi; 
    }

    static suffstats_t sample_from_prior(hypers_t * hps, rng_t & rng) {
        suffstats_t ss; 
        for(int k = 0; k < hps->comp_k; ++k) { 
            ss.mu.push_back(uniform(EPSILON, 1.0 - EPSILON, rng)); 
            ss.var.push_back(chi2_sample(CHI_VAL, rng) * hps->var_scale + EPSILON); 
            assert (ss.var[k] < 1e10); 
        }

        ss.pi = symmetric_dirichlet_sample(hps->comp_k, 
                                           hps->dir_alpha, rng); 


        return ss; 
        
    }

    static void ss_add(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 

    }

    static void ss_rem(suffstats_t * ss, hypers_t * hps, value_t val, 
                       dppos_t dp_pos) { 
    }


    static float post_pred(suffstats_t * ss, hypers_t * hps, value_t val) {

        // Gets a single row of observations
        return data_prob_mm(val, ss); 
        
    }

    static float score_prior(suffstats_t * ss, hypers_t * hps) { 
        int COMP_K = hps->comp_k; 
        float score = 0.0; 

        for(int k = 0; k < COMP_K; ++k) { 
            if ((ss->mu[k] <= EPSILON) |  (ss->mu[k] >= (1.0 - EPSILON))) { 
                return -std::numeric_limits<float>::infinity(); 
            }
            if ((ss->var[k] / hps->var_scale) < EPSILON) { 
                return -std::numeric_limits<float>::infinity(); 
            }
            score += log_chi2_dist(ss->var[k] / hps->var_scale, CHI_VAL); 

        }
        score += log_symmetric_dir_dist(ss->pi, hps->dir_alpha); 

        return score; 
    }
    
    static float data_prob_mm(const value_t & val, 
                            suffstats_t * ss) 
    {
        int N = val.size(); 
        int K = ss->pi.size(); 
        
        float tot_score = 0.0; 
        for(int n = 0; n < N; ++n) { 
            float score = -1e80; 
            for(int k = 0; k < K; ++k) { 
                float mu = ss->mu[k]; 
                float pi = ss->pi[k]; 
                float sigmasq = ss->var[k]; 
                float s = log_norm_dist(val[n], mu, sigmasq); 
                s += MYLOG(pi); 
                score = log_sum_exp(score, s); 
            }
            tot_score += score; 

        }
        return tot_score; 
    }


    template<typename RandomAccessIterator, typename RandomAccessIterator2> 
    static float score_likelihood(suffstats_t * ss, hypers_t * hps, 
                                  RandomAccessIterator data, 
                                  RandomAccessIterator2 observed, 
                                  const std::vector<dppos_t> & dppos)
    {
        float score = 0.0; 
        for(auto pos : dppos) { 
            if(observed[pos]) { 
                score += post_pred(ss, hps, data[pos]); 
            }
        }
        return score; 

    }

    template<typename RandomAccessIterator, typename RandomAccessIterator2>
    static float score(suffstats_t * ss, hypers_t * hps, 
                       RandomAccessIterator data,
                       RandomAccessIterator2 observed, 
                       const std::vector<dppos_t> & dppos) { 

        float prior_score = score_prior(ss, hps); 
        float likelihood_score = score_likelihood(ss, hps, data, observed, dppos); 
        return prior_score + likelihood_score; 

    }

    static hypers_t bp_dict_to_hps(bp::dict & hps) { 
        hypers_t hp; 
        hp.comp_k = bp::extract<int>(hps["comp_k"]); 
        hp.dir_alpha = bp::extract<float>(hps["dir_alpha"]);
        hp.var_scale = bp::extract<float>(hps["var_scale"]);
        return hp; 

    }
    static bp::dict hps_to_bp_dict(const hypers_t  & hps) {
        bp::dict hp; 
        hp["comp_k"] = hps.comp_k; 
        hp["dir_alpha"] = hps.dir_alpha; 
        hp["var_scale"] = hps.var_scale; 
        return hp; 

    }
    static bp::list vect_f32_to_list(const std::vector<float> & v) {
        bp::list l; 
        for(float x : v) { 
            l.append(x); 
        }
        return l ; 
    }
    static std::vector<float> list_to_vect_f32(bp::list l) { 
        std::vector<float> v; 
        for(int i = 0; i < bp::len(l); ++i) { 
            v.push_back(bp::extract<float>(l[i])); 
        }
        return v; 

    }
    static bp::dict ss_to_dict(suffstats_t * ss) { 
        bp::dict d; 
        d["mu"] = vect_f32_to_list(ss->mu); 
        d["var"] = vect_f32_to_list(ss->var); 
        d["pi"] = vect_f32_to_list(ss->pi); 
        return d; 
    }

    static void ss_from_dict(suffstats_t * ss, bp::dict v) { 
        ss->mu = list_to_vect_f32(bp::extract<bp::list>(v["mu"])); 
        ss->var = list_to_vect_f32(bp::extract<bp::list>(v["var"])); 
        ss->pi = list_to_vect_f32(bp::extract<bp::list>(v["pi"])); 

    }

}; 

}

#endif
