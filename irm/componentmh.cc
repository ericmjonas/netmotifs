#include "componentmh.h"
#include "kernels.h"

namespace irm { 


template<>
void continuous_mh_sample_exec<BetaBernoulliNonConj>
(rng_t & rng, int iters, float min, float max,  
 BetaBernoulliNonConj::suffstats_t * ss, 
 BetaBernoulliNonConj::hypers_t * hps, 
 std::vector<BetaBernoulliNonConj::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){

    auto p = continuous_mh_sample<float>(ss->p, 
                                 [&ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->p = x; 
                                     return BetaBernoulliNonConj::score(ss, hps, data, dppos)/temp;
                          }, 
                                         rng, iters, min, max); 
    
    ss->p = p; 

}

template<>
void continuous_mh_sample_exec<LogisticDistance>
(rng_t & rng, int iters, float min, float max,  
 LogisticDistance::suffstats_t * ss, 
 LogisticDistance::hypers_t * hps, 
 std::vector<LogisticDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    auto mu = continuous_mh_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LogisticDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, 
                                  rng, iters, min, max); 
    
    ss->mu = mu; 

    auto lambda = continuous_mh_sample<float>(ss->lambda, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->lambda = x; 
                                          return LogisticDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, 
                                      rng, iters, min, max); 
    
    ss->lambda = lambda; 

}

template<>
void continuous_mh_sample_exec<SigmoidDistance>
(rng_t & rng, int iters, float min, float max,  
 SigmoidDistance::suffstats_t * ss, 
 SigmoidDistance::hypers_t * hps, 
 std::vector<SigmoidDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){

    auto mu = continuous_mh_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->mu = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                          }, 
                          rng, iters, min, max); 
    
    ss->mu = mu; 


    auto lambda = continuous_mh_sample<float>(ss->lambda, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->lambda = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                          }, 
                          rng, iters, min, max); 
    
    ss->lambda = lambda; 

}


template<> void continuous_mh_sample_exec<LinearDistance>
(rng_t & rng, int iters, float min, float max,  
 LinearDistance::suffstats_t * ss, 
 LinearDistance::hypers_t * hps, 
 std::vector<LinearDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    auto mu = continuous_mh_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LinearDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, 
                                  rng, iters, min, max); 
    
    ss->mu = mu; 

    auto p = continuous_mh_sample<float>(ss->p, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p = x; 
                                          return LinearDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, 
                                         rng, iters, min, max); 
    
    ss->p = p; 

}

template<>
void continuous_mh_sample_exec<BetaBernoulli>
(rng_t & rng, int iters, float min, float max,  
 BetaBernoulli::suffstats_t * ss, 
 BetaBernoulli::hypers_t * hps, 
 std::vector<BetaBernoulli::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    // Doesn't do anything because the model is conjugate

}

template<>
void continuous_mh_sample_exec<GammaPoisson>
(rng_t & rng, int iters, float min, float max,  
 GammaPoisson::suffstats_t * ss, 
 GammaPoisson::hypers_t * hps, 
 std::vector<GammaPoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    // Doesn't do anything because the model is conjugate

}


}
