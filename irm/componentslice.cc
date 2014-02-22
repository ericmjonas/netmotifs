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
void slice_sample_exec<LogisticDistanceFixedLambda>
(rng_t & rng, float width, 
 LogisticDistanceFixedLambda::suffstats_t * ss, 
 LogisticDistanceFixedLambda::hypers_t * hps, 
 std::vector<LogisticDistanceFixedLambda::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){
    if (width == 0.0) {
        width = hps->mu_hp*2.0; 
    } else {
        std::cout << "Slice width manually set to " << width 
                  << " (automatic would have been " 
                  << hps->mu_hp/4.0 << ")" << std::endl; 


    }
    try { 
        auto mu = slice_sample2_double(
                                       [ss, &hps, data, &dppos, temp](float x) -> float{
                                           ss->mu = x; 
                                           return LogisticDistanceFixedLambda::score(ss, hps, data, 
                                                                     dppos) /temp;
                                       }, ss->mu, width, rng); 
        
        ss->mu = mu; 
    } catch (std::exception & e){ 
        std::cout << "ss->mu=" << ss->mu 
                  << " hps->mu_hp=" << hps->mu_hp 
                  << " ss->p_scale=" << ss->p_scale 
                  << " hps->lambda = " << hps->lambda
                  << " hps->p_scale_alpha_hp = " << hps->p_scale_alpha_hp 
                  << " hps->p_scale_beta_hp = " << hps->p_scale_beta_hp 
                  << std::endl; 
        throw; 
    }
    try { 
    auto p_scale = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->p_scale = x; 
                                          return LogisticDistanceFixedLambda::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->p_scale, 
                                      0.1, rng); 
    
    ss->p_scale = p_scale; 
    } catch (std::exception & e){ 
        std::cout << "ss->p_scale=" << ss->p_scale  << std::endl; 
        throw; 
    }

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
void slice_sample_exec<NormalInverseChiSq>
(rng_t & rng, float width, 
 NormalInverseChiSq::suffstats_t * ss, 
 NormalInverseChiSq::hypers_t * hps, 
 std::vector<NormalInverseChiSq::value_t>::iterator data,
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

template<> void slice_sample_exec<ExponentialDistancePoisson>
(rng_t & rng, float width, 
 ExponentialDistancePoisson::suffstats_t * ss, 
 ExponentialDistancePoisson::hypers_t * hps, 
 std::vector<ExponentialDistancePoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    float mu_width = width; 
    float rate_scale_width = width; 
    if (width == 0.0) {
        mu_width = hps->mu_hp*4.0; 
    }
    if (width == 0.0) {
        rate_scale_width = hps->rate_scale_hp*4.0; 
    }
    //std::cout << "EDP : slice sampling mu, mu_width=" << mu_width << std::endl; 
    auto mu = slice_sample2_double(
                            [ss, &hps, data, &dppos, temp](float x) -> float{
                                ss->mu = x; 
                                if(x > 1e50) { 
                                    throw std::runtime_error("mu the hell do you think you are?"); 
                                }

                                return ExponentialDistancePoisson::score(ss, hps, data, 
                                                                         dppos) /temp;
                            }, ss->mu, mu_width, rng); 
    
    ss->mu = mu; 
    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    //std::cout << "EDP : slice sampling rate_scale, rate_scale_width=" << rate_scale_width << std::endl; 

    auto rate_scale = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->rate_scale = x; 
                                          return ExponentialDistancePoisson::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->rate_scale, rate_scale_width, rng); 

    
    ss->rate_scale = rate_scale; 

}


template<> void slice_sample_exec<LogisticDistancePoisson>
(rng_t & rng, float width, 
 LogisticDistancePoisson::suffstats_t * ss, 
 LogisticDistancePoisson::hypers_t * hps, 
 std::vector<LogisticDistancePoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    float mu_width = width; 
    float rate_scale_width = width; 
    if (width == 0.0) {
        mu_width = hps->mu_hp*4.0; 
    }
    if (width == 0.0) {
        rate_scale_width = hps->rate_scale_hp*4.0; 
    }
    //std::cout << "EDP : slice sampling mu, mu_width=" << mu_width << std::endl; 
    auto mu = slice_sample2_double(
                            [ss, &hps, data, &dppos, temp](float x) -> float{
                                ss->mu = x; 
                                if(x > 1e50) { 
                                    throw std::runtime_error("mu the hell do you think you are?"); 
                                }

                                return LogisticDistancePoisson::score(ss, hps, data, 
                                                                         dppos) /temp;
                            }, ss->mu, mu_width, rng); 
    
    ss->mu = mu; 
    // the width for this is always 0.1 because we're always sampling 
    // on [0, 1]
    //std::cout << "EDP : slice sampling rate_scale, rate_scale_width=" << rate_scale_width << std::endl; 

    auto rate_scale = slice_sample2_double(
                                      [ss, &hps, data, &dppos, temp](float x) -> float{
                                          ss->rate_scale = x; 
                                          return LogisticDistancePoisson::score(ss, hps, data, 
                                                                         dppos)/temp;
                                      }, ss->rate_scale, rate_scale_width, rng); 

    
    ss->rate_scale = rate_scale; 

}


template<>
void slice_sample_exec<MixtureModelDistribution>
(rng_t & rng, float width, 
 MixtureModelDistribution::suffstats_t * ss, 
 MixtureModelDistribution::hypers_t * hps, 
 std::vector<MixtureModelDistribution::value_t>::iterator data,
 const std::vector<dppos_t> & dppos,
 float temp){

    if (width == 0.0) { 
        width = 0.2; 
    }
    assert(hps->comp_k == ss->var.size()); 
    if(boost::math::isnan(MixtureModelDistribution::score(ss, hps, data, 
                                               dppos))) {
        std::cout << "beginning slice sampling from nan" << std::endl; 
        for(int k = 0; k < hps->comp_k; ++k) { 
            std::cout << "k=" << k 
                      << " mu=" << ss->mu[k] 
                      << " var=" << ss->var[k] << std::endl; 
        }
    }
    for(int k = 0; k < hps->comp_k; ++k) { 
        // std::cout << "before var[" << k << "]= " <<  ss->var[k] 
        //           << " mu=" << ss->mu[k] << std::endl; 

        auto mu = slice_sample2_double(
                                       [ss, k, &hps, data, &dppos, temp](float x) -> float{
                                           ss->mu[k] = x; 
                                           return MixtureModelDistribution::score(ss, hps, data, 
                                                                                  dppos) /temp;
                                       }, ss->mu[k], width, rng); 
    
        ss->mu[k] = mu; 
        // std::cout << "before var[" << k << "]= " <<  ss->var[k] << std::endl; 
        
        auto var = slice_sample2_double(
                                       [ss, k, &hps, data, &dppos, temp](float x) -> float{
                                           ss->var[k] = x; 
                                           return MixtureModelDistribution::score(ss, hps, data, 
                                                                                  dppos) /temp;
                                       }, ss->var[k], width, rng); 
        ss->var[k] = var; 
        // std::cout << "var[" << k << "]= " <<  var << std::endl; 



    }

    // sample dirichlet and do MH
    auto oldpi = ss->pi; 

    float oldscore = MixtureModelDistribution::score(ss, hps, data, dppos)/temp; 
    ss->pi = symmetric_dirichlet_sample(hps->comp_k, hps->dir_alpha, rng); 
    float newscore = MixtureModelDistribution::score(ss, hps, data, dppos)/temp; 
    if (uniform_01(rng) < exp(newscore - oldscore)) { 
        // accept
    } else { 
        ss->pi = oldpi; 
    }
        
    
    
}



}
