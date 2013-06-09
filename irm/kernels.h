#ifndef __IRM_KERNELS_H__
#define __IRM_KERNELS_H__
#include <functional>

namespace irm { 

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

    
    while (P(x_l) > uprime) {
        x_l -= w; 
    }
    while (P(x_r) > uprime) { 
        x_r += w; 
    }
    while(true) { 
        T xprime = uniform(x_l, x_r, rng); 

        if(P(xprime) > uprime) { 
            return xprime; 
        }
        if(xprime > x) { 
            x_r = xprime; 
        } else { 
            x_l = xprime; 
        }
    }


}




}


#endif
