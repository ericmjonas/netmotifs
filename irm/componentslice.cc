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
 std::vector<BetaBernoulliNonConj::value_t>::iterator data){
    // std::cout << "hps->alpha=" << hps->alpha 
    //           << " hps->beta=" << hps->beta << std::endl;
    auto p = slice_sample<float>(ss->p, 
                                 [&ss, &hps, data](float x) -> float{
                                     ss->p = x; 
                                     return BetaBernoulliNonConj::score(ss, hps, data);
                          }, 
                          rng, width); 
    
    ss->p = p; 

}

template<>
void slice_sample_exec<LogisticDistance>
(rng_t & rng, float width, 
 LogisticDistance::suffstats_t * ss, 
 LogisticDistance::hypers_t * hps, 
 std::vector<LogisticDistance::value_t>::iterator data){
    auto mu = slice_sample<float>(ss->mu, 
                                 [&ss, &hps, data](float x) -> float{
                                     ss->mu = x; 
                                     return LogisticDistance::score(ss, hps, data);
                          }, 
                          rng, width); 
    
    ss->mu = mu; 


    auto lambda = slice_sample<float>(ss->lambda, 
                                 [&ss, &hps, data](float x) -> float{
                                     ss->lambda = x; 
                                     return LogisticDistance::score(ss, hps, data);
                          }, 
                          rng, width); 
    
    ss->lambda = lambda; 

}

}
