
#include <iostream>
#include <map>
#include <inttypes.h>
#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <vector>

#include "util.h"
#include "componentmodels.h"
#include "kernels.h"

#pragma once 

namespace irm { 
/*
  
  
*/ 

template<typename T>
void continuous_mh_sample_exec
(rng_t & rng, int iters, float min, float max, 
 typename T::suffstats_t * ss, 
 typename T::hypers_t * hps, 
 typename std::vector<typename T::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp)
{

    throw std::runtime_error("slice sampler not implemented for this component model"); 
}

template<>
void continuous_mh_sample_exec<BetaBernoulliNonConj>
(rng_t & rng, int iters, float min, float max,  
 BetaBernoulliNonConj::suffstats_t * ss, 
 BetaBernoulliNonConj::hypers_t * hps, 
 std::vector<BetaBernoulliNonConj::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void continuous_mh_sample_exec<LogisticDistance>
(rng_t & rng, int iters, float min, float max,  
 LogisticDistance::suffstats_t * ss, 
 LogisticDistance::hypers_t * hps, 
 std::vector<LogisticDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);


template<>
void continuous_mh_sample_exec<SigmoidDistance>
(rng_t & rng, int iters, float min, float max,  
 SigmoidDistance::suffstats_t * ss, 
 SigmoidDistance::hypers_t * hps, 
 std::vector<SigmoidDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void continuous_mh_sample_exec<LinearDistance>
(rng_t & rng, int iters, float min, float max,  
 LinearDistance::suffstats_t * ss, 
 LinearDistance::hypers_t * hps, 
 std::vector<LinearDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);

template<>
void continuous_mh_sample_exec<BetaBernoulli>
(rng_t & rng, int iters, float min, float max,  
 BetaBernoulli::suffstats_t * ss, 
 BetaBernoulli::hypers_t * hps, 
 std::vector<BetaBernoulli::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);

template<>
void continuous_mh_sample_exec<GammaPoisson>
(rng_t & rng, int iters, float min, float max,  
 GammaPoisson::suffstats_t * ss, 
 GammaPoisson::hypers_t * hps, 
 std::vector<GammaPoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);


}

