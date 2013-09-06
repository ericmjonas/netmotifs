#ifndef __IRM_KERNELS_H__
#define __IRM_KERNELS_H__
#include <functional>
#include "slicesample.h"

namespace irm { 

const int LOOPMAX = 100; 

template<typename T>
T slice_sample(T x, 
               std::function<float(T)> P, rng_t &  rng, 
               float w) 
{
    float Pstar = P(x); 

    T uprime = logf(uniform_01(rng)) + Pstar; 
    //T uprime = uniform(0, Pstar, rng); 
    T x_l, x_r; 
    // Create initial interval 
    float r = uniform_01(rng); 
    x_l = x - r*w; 
    x_r = x + (1-r) * w; 

    int loopcnt =0; 
    while ((P(x_l) > uprime) and (loopcnt < LOOPMAX)) {
        x_l -= w; 
        loopcnt++; 
    }

    if (loopcnt == LOOPMAX) { 
        std::cerr << "Warning, slice sampling x_l loop reached LOOPMAX" 
                  << std::endl; 
    }

    loopcnt = 0; 
    while ((P(x_r) > uprime) and (loopcnt < LOOPMAX)) { 
        x_r += w; 
        loopcnt++; 
    }

    if (loopcnt == LOOPMAX) { 
        std::cerr << "Warning, slice sampling x_l loop reached LOOPMAX" 
                  << std::endl; 
    }

    loopcnt = 0; 
    while(loopcnt < LOOPMAX) { 
        T xprime = uniform(x_l, x_r, rng); 

        if(P(xprime) > uprime) { 
            return xprime; 
        }
        if(xprime > x) { 
            x_r = xprime; 
        } else { 
            x_l = xprime; 
        }
        
        loopcnt++; 
    }

    std::cerr << "WARNING slice sampling failed " 
              << " x_r =" << x_r 
              << " x_l = " << x_l 
              << " uprime=" << uprime 
              << " Pstar=" << Pstar 
              << " x=" << x << std::endl; 
    return x; 



}

template<typename T>
T continuous_mh_sample(T x, 
                       std::function<float(T)> P, rng_t &  rng, 
                       int ITERS, float LOG_SCALE_MIN, float LOG_SCALE_MAX)
{
    /* run ITERS of a mh where the width is drawn from 
       a distribution exp(unif(LOG_SCALE_MIN, LOG_SCALE_MAX))
       
       x: initial value
       P: function to evaluate log score
       ITERS :  how many internal loops 

    */ 
    T cur_x = x; 
    float cur_score = P(cur_x); 
    for(int i = 0; i < ITERS; ++i) { 
        float width = pow(uniform(LOG_SCALE_MIN, LOG_SCALE_MAX, rng), 10); 
        T new_x = normal_sample(cur_x, width, rng); 
        float new_score = P(new_x); 

        if(uniform_01(rng) < MYEXP(new_score - cur_score)) { 
            cur_score = new_score; 
            cur_x = new_x; 
        } else { 
            
        }
    }
    
    return cur_x; 



}



}


#endif
