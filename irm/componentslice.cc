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

    auto p = slice_sample2_double(
                                 [&ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->p = x; 
                                     return BetaBernoulliNonConj::score(ss, hps, data, dppos)/temp;
                                 }, ss->p, width, rng); 
    
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
        width = hps->mu_hp*2.0; 
    } else {
        std::cout << "Slice width manually set to " << width 
                  << " (automatic would have been " 
                  << hps->mu_hp/4.0 << ")" << std::endl; 


    }
    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LogisticDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, ss->mu, width, rng); 
    
    ss->mu = mu; 

    auto lambda = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->lambda = x; 
                                          return LogisticDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->lambda, 
                                      width, rng); 
    
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

    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->mu = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                                  }, ss->mu, width, rng); 
    
    ss->mu = mu; 


    auto lambda = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                     ss->lambda = x; 
                                     return SigmoidDistance::score(ss, hps, data, 
                                                                    dppos)/temp;
                                      }, ss->lambda, width, rng); 

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

    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LinearDistance::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, ss->mu, width, rng); 
    
    ss->mu = mu; 

    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    auto p = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p = x; 
                                          return LinearDistance::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->p, 0.5, rng); 

    
    ss->p = p; 

}

template<>
void slice_sample_exec<BetaBernoulli>
(rng_t & rng, float width, 
 BetaBernoulli::suffstats_t * ss, 
 BetaBernoulli::hypers_t * hps, 
 std::vector<BetaBernoulli::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    // Doesn't do anything because the model is conjugate

}

template<>
void slice_sample_exec<GammaPoisson>
(rng_t & rng, float width, 
 GammaPoisson::suffstats_t * ss, 
 GammaPoisson::hypers_t * hps, 
 std::vector<GammaPoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp){
    // Doesn't do anything because the model is conjugate

}

template<> void slice_sample_exec<NormalDistanceFixedWidth>
(rng_t & rng, float width, 
 NormalDistanceFixedWidth::suffstats_t * ss, 
 NormalDistanceFixedWidth::hypers_t * hps, 
 std::vector<NormalDistanceFixedWidth::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    if (width == 0.0) {
        width = hps->mu_hp*2.0; 
    }

    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return NormalDistanceFixedWidth::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, ss->mu, width, rng);
    
    ss->mu = mu; 

    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    auto p = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p = x; 
                                          return NormalDistanceFixedWidth::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->p, 0.5, rng); 
    
    ss->p = p; 

}

template<> void slice_sample_exec<SquareDistanceBump>
(rng_t & rng, float width, 
 SquareDistanceBump::suffstats_t * ss, 
 SquareDistanceBump::hypers_t * hps, 
 std::vector<SquareDistanceBump::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    if (width == 0.0) {
        width = hps->mu_hp* 2.0; 
    }

    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return SquareDistanceBump::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, ss->mu,  width, rng); 
    
    ss->mu = mu; 

    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    auto p = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p = x; 
                                          return SquareDistanceBump::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->p, 0.5, rng); 

    
    ss->p = p; 

}

template<> void slice_sample_exec<LinearDistancePoisson>
(rng_t & rng, float width, 
 LinearDistancePoisson::suffstats_t * ss, 
 LinearDistancePoisson::hypers_t * hps, 
 std::vector<LinearDistancePoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    float mu_width = width; 
    float rate_width = width; 
    if (width == 0.0) {
        mu_width = hps->mu_hp/4.0; 
    }
    if (width == 0.0) {
        rate_width = hps->mu_hp/4.0; 
    }

    auto mu = slice_sample2_double(
                                  [ss, &hps, data, &dppos, temp](float x) -> float{
                                      ss->mu = x; 
                                      return LinearDistancePoisson::score(ss, hps, data, 
                                                                     dppos) /temp;
                                  }, ss->mu, mu_width, rng); 
    
    ss->mu = mu; 

    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    auto rate = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->rate = x; 
                                          return LinearDistancePoisson::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->rate, rate_width, rng); 

    
    ss->rate = rate; 

}


}
