#include <iostream>
#include "util.h"
#include <boost/math/distributions.hpp>

int main()
{
    using namespace irm; 

    float mu_hp = 10.0; 
    float sum = 0.0; 
    int N = 100000; 
    rng_t rng; 
    for(int i =0; i < N; ++i) { 
    
        float r1 = uniform_01(rng); 
        
        boost::math::exponential_distribution<> mu_dist(1.0/mu_hp);
        float mu = quantile(mu_dist, r1); 
        sum += mu; 
    }
    std::cout << "mean=" << sum / N  << std::endl; 


}
