#include <functional>
#include "util.h"

#pragma once

namespace irm { 


std::pair<float, float> 
create_interval(std::function<float(float)> f, float x_0, float y, float w, 
                int m, rng_t &  rng); 

std::pair<float, float> 
create_interval_double(std::function<float(float)> f, float x_0, float y, float w, 
                       int p, rng_t &  rng);
bool double_accept(std::function<float(float)> f, float x_0, 
                    float x_1, float y, float w, 
                   std::pair<float, float>  interval); 

float shrink(std::pair<float, float> LR, 
             std::function<float(float)> f, float x_0, float y, float w, 
             bool use_accept, rng_t &  rng);

float slice_sample2(std::function<float(float)> f, float x_0, 
                   float w, rng_t &  rng) ;

float slice_sample2_double(std::function<float(float)> f, float x_0, 
                          float w, rng_t &  rng) ; 


}
