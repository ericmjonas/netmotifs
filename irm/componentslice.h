#ifndef __IRM_COMPSLICE_H__
#define __IRM_COMPSLICE_H__

#include <iostream>
#include <map>
#include <inttypes.h>
#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <vector>

#include "util.h"
#include "componentmodels.h"
#include "kernels.h"

namespace irm { 
/*
  Slice sampling for the non-conjugate hyperparameter case
  
  for each component, we get the data associated with it, and we 
  then slice sample to do a posterior update of the latent
  parameter 
  
  
*/ 

template<typename T>
void slice_sample_exec
(rng_t & rng, float width, 
 typename T::suffstats_t * ss, 
 typename T::hypers_t * hps, 
 typename std::vector<typename T::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp)
{

    throw std::runtime_error("slice sampler not implemented for this component model"); 
}

template<>
void slice_sample_exec<BetaBernoulliNonConj>
(rng_t & rng, float width, 
 BetaBernoulliNonConj::suffstats_t * ss, 
 BetaBernoulliNonConj::hypers_t * hps, 
 std::vector<BetaBernoulliNonConj::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void slice_sample_exec<LogisticDistance>
(rng_t & rng, float width, 
 LogisticDistance::suffstats_t * ss, 
 LogisticDistance::hypers_t * hps, 
 std::vector<LogisticDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);


template<>
void slice_sample_exec<SigmoidDistance>
(rng_t & rng, float width, 
 SigmoidDistance::suffstats_t * ss, 
 SigmoidDistance::hypers_t * hps, 
 std::vector<SigmoidDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void slice_sample_exec<LinearDistance>
(rng_t & rng, float width, 
 LinearDistance::suffstats_t * ss, 
 LinearDistance::hypers_t * hps, 
 std::vector<LinearDistance::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);

template<>
void slice_sample_exec<BetaBernoulli>
(rng_t & rng, float width, 
 BetaBernoulli::suffstats_t * ss, 
 BetaBernoulli::hypers_t * hps, 
 std::vector<BetaBernoulli::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);

template<>
void slice_sample_exec<GammaPoisson>
(rng_t & rng, float width, 
 GammaPoisson::suffstats_t * ss, 
 GammaPoisson::hypers_t * hps, 
 std::vector<GammaPoisson::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void slice_sample_exec<NormalDistanceFixedWidth>
(rng_t & rng, float width, 
 NormalDistanceFixedWidth::suffstats_t * ss, 
 NormalDistanceFixedWidth::hypers_t * hps, 
 std::vector<NormalDistanceFixedWidth::value_t>::iterator data,
 const std::vector<dppos_t> & dppos, 
 float temp);


template<>
void slice_sample_exec<SquareDistanceBump>
(rng_t & rng, float width, 
 SquareDistanceBump::suffstats_t * ss, 
 SquareDistanceBump::hypers_t * hps, 
 std::vector<SquareDistanceBump::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);


template<>
void slice_sample_exec<ExponentialDistancePoisson>
(rng_t & rng, float width, 
 ExponentialDistancePoisson::suffstats_t * ss, 
 ExponentialDistancePoisson::hypers_t * hps, 
 std::vector<ExponentialDistancePoisson::value_t>::iterator data, 
 const std::vector<dppos_t> & dppos,
 float temp);

}

#endif
