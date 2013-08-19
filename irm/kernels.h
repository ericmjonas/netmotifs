#ifndef __IRM_KERNELS_H__
#define __IRM_KERNELS_H__
#include <functional>

namespace irm { 

const int LOOPMAX = 1000; 

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
    if(loopcnt == LOOPMAX) { 
        std::cout << "x_l expansion failed " << x_l << std::endl; 
        return x; 
    }

    loopcnt = 0; 
    while ((P(x_r) > uprime) and (loopcnt < LOOPMAX)) { 
        x_r += w; 
        loopcnt++; 
    }

    if(loopcnt == LOOPMAX) { 
        std::cout << "x_r expansion failed " << x_r << std::endl; 
        return x; 
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

    std::cout << "slice sampling failed failed " 
              << " x_r =" << x_r 
              << " x_l = " << x_l 
              << " uprime=" << uprime 
              << " Pstar=" << Pstar 
              << " x=" << x << std::endl; 
    return x; 



}




}


#endif
