#include "slicesample.h"

namespace irm { 

const int SLICE_M = 100; 
const int SLICE_P = 20; 
const int SLICE_MAX_SHRINK = 100; 

std::pair<float, float> 
create_interval(std::function<float(float)> f, float x_0, float y, float w, 
                int m, rng_t &  rng)
{
    float U = uniform_01(rng); 
    float L = x_0 - w * U; 
    float R = L + w; 
    float V = uniform_01(rng); 
    int J = floor(m * V); 
    int K = (m- 1) - J; 

    while ((J > 0) and (y < f(L))) { 
        L -= w; 
        J--; 
    }
    
    while ((K > 0) and (y < f(R))) { 
        R += w; 
        K -=1; 
    }
    
    if (J == 0) { 
        std::cerr << "Warning: L expansion reached limit" << std::endl; 
    }
    if (K == 0) { 
        std::cerr << "Warning: R expansion reached limit" << std::endl; 
    }

    return std::make_pair(L, R); 
    
}

std::pair<float, float> 
create_interval_double(std::function<float(float)> f, float x_0, float y, float w, 
                       int p, rng_t &  rng)
{
    float U = uniform_01(rng); 
    float L = x_0 - w * U; 
    float R = L + w; 
    int K = p; 

    float last_f_l = f(L); 
    float last_f_r = f(R); 
    while ((K > 0) and ((y < last_f_l ) or (y < last_f_r))) { 
        float V = uniform_01(rng); 
        if (V < 0.5) { 
            
            L -= (R -L); 
            last_f_l = f(L); 
            
        } else { 
            R += (R - L); 
                last_f_r = f(R); 
        }
        
    }

    if (K == 0) { 
        std::cerr << "Warning: double expansion reached limit" << std::endl; 
    }

    return std::make_pair(L, R); 
    
}

bool double_accept(std::function<float(float)> f, float x_0, 
                    float x_1, float y, float w, 
                    std::pair<float, float>  interval)
{
    float Lhat = interval.first; 
    float Rhat = interval.second; 
    bool D = false; 
    while ((Rhat - Lhat)  > 1.1*w) { 
        float M = (Lhat + Rhat) / 2.0; 
        if (((x_0 < M) && (x_1 >= M)) ||
            ((x_0 >=M) && (x_1 < M))) { 
            D = true; 
        }
        if (x_1 < M) {
            Rhat = M; 
        } else { 
            Lhat = M; 
        }
        if (D && ((y >= f(Lhat)) && (y >= f(Rhat)))) { 
            return false; 
            
        }
    }
    return true; 
}

float shrink(std::pair<float, float> LR, 
             std::function<float(float)> f, float x_0, float y, float w, 
             bool use_accept, rng_t &  rng)
{
    float Lbar = LR.first; 
    float Rbar = LR.second; 
    int shrink = SLICE_MAX_SHRINK; 
    while(shrink > 0) { 
        float U = uniform_01(rng); 
        float x_1 = Lbar + U * (Rbar - Lbar); 
        if (use_accept) { 
            if ((y < f(x_1) ) && double_accept(f, x_0, x_1, y, w, 
                                               LR)) {
                return x_1; 
            }
        } else { 
            if (y < f(x_1)) { 
                return x_1;
            }
        }
        
        if (x_1 < x_0) { 
            Lbar = x_1; 
        } else { 
            Rbar = x_1; 
        }

        shrink--; 
        
    }
    return x_0; // no change
    std::cerr << "Shrink iters exceeded SLICE_MAX_SHRINK" << std::endl; 
}

/* second, more-correct implementaiton, now with doubling
 */ 


float slice_sample2(std::function<float(float)> f, float x_0, 
                   float w, rng_t &  rng) 
{

    float y = logf(uniform_01(rng)) + f(x_0); 
    
    auto interval = create_interval(f, x_0, y, w, SLICE_M, rng); 
    return shrink(interval, f, x_0, y, w, false, rng); 

}


float slice_sample2_double(std::function<float(float)> f, float x_0, 
                          float w, rng_t &  rng) 
{
    float y = logf(uniform_01(rng)) + f(x_0); 
    if(std::isinf(y)) { 
        throw std::runtime_error("Beginning slice sampling from impossible x_0"); 
    }
    auto interval = create_interval_double(f, x_0, y, w, SLICE_P, rng); 
    return shrink(interval, f, x_0, y, w, true, rng); 

}




}
