#include "componentslice.h"
#include "kernels.h"

namespace irm { 

// void slice_sample_exec
// (rng_t & rng, float width, 
//  BetaBernoulliNonConj::suffstats_t * ss, 
//  BetaBernoulliNonConj::hypers_t * hps, 
//  std::vector<BetaBernoulliNonConj::value_t>::iterator data)
// {
//     auto p = slice_sample<float>(ss->p, 
//                                  [&ss, &hps, data](float x) -> float{
//                                      ss->p = x; 
//                                      return BetaBernoulliNonConj::score(ss, hps, data);
//                           }, 
//                           rng, width); 
    
//     ss->p = p; 

// }

// template<>
// void slice_sample_exec<BetaBernoulliNonConj>
// (rng_t & rng, float width, 
//  BetaBernoulliNonConj::suffstats_t * ss, 
//  BetaBernoulliNonConj::hypers_t * hps, 
//  std::vector<BetaBernoulliNonConj::value_t>::iterator data){

//     auto p = slice_sample<float>(ss->p, 
//                                  [&ss, &hps, data](float x) -> float{
//                                      ss->p = x; 
//                                      return BetaBernoulliNonConj::score(ss, hps, data);
//                           }, 
//                           rng, width); 
    
//     ss->p = p; 

// }

template<>
void slice_sample_exec<BetaBernoulliNonConj>
(rng_t & rng, float width, 
 BetaBernoulliNonConj::suffstats_t * ss, 
 BetaBernoulliNonConj::hypers_t * hps, 
 std::vector<BetaBernoulliNonConj::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    if (width == 0.0) { 
        width = 0.1; 
    }

    auto p = slice_sample<float>(ss->p, 
                                 [&ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->p = x; 
                                     return BetaBernoulliNonConj::score(ss, hps, data, dppos)/temp;
                          }, 
                          rng, width); 
    
    ss->p = p; 

}

template<>
void slice_sample_exec<LogisticDistance>
(rng_t & rng, float width, 
 LogisticDistance::suffstats_t * ss, 
 LogisticDistance::hypers_t * hps, 
 std::vector<LogisticDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){
    if (width == 0.0) {
        width = hps->mu_hp/4.0; 
    } else {
        std::cout << "Slice width manually set to " << width 
                  << " (automatic would have been " 
                  << hps->mu_hp/4.0 << ")" << std::endl; 


    }
    auto mu = slice_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LogisticDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, 
                                  rng, width); 
    
    ss->mu = mu; 

    auto lambda = slice_sample<float>(ss->lambda, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->lambda = x; 
                                          return LogisticDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, 
                                      rng, width); 
    
    ss->lambda = lambda; 

}

template<>
void slice_sample_exec<SigmoidDistance>
(rng_t & rng, float width, 
 SigmoidDistance::suffstats_t * ss, 
 SigmoidDistance::hypers_t * hps, 
 std::vector<SigmoidDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    if (width == 0.0) {
        width = hps->mu_hp/4.0; 
    }

    auto mu = slice_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->mu = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                          }, 
                          rng, width); 
    
    ss->mu = mu; 


    auto lambda = slice_sample<float>(ss->lambda, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->lambda = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                          }, 
                          rng, width); 
    
    ss->lambda = lambda; 

}


template<> void slice_sample_exec<LinearDistance>
(rng_t & rng, float width, 
 LinearDistance::suffstats_t * ss, 
 LinearDistance::hypers_t * hps, 
 std::vector<LinearDistance::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    if (width == 0.0) {
        width = hps->mu_hp/4.0; 
    }

    auto mu = slice_sample<float>(ss->mu, 
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LinearDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, 
                                  rng, width); 
    
    ss->mu = mu; 

    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    auto p = slice_sample<float>(ss->p, 
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p = x; 
                                          return LinearDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, 
                                      rng, 0.1); 
    
    ss->p = p; 

}


}
