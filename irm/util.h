#ifndef __IRM_UTIL_H__
#define __IRM_UTIL_H__

#include <list>
#include <set>
#include <vector>
#include <array>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <stdlib.h>
#include "fastonebigheader.h"

#include "group_coords.h"


#ifdef USE_LOGEXP_APPROX
#define MYLOG fasterlog
#define MYEXP fasterexp

#endif 

#ifndef USE_LOGEXP_APPROX
#define MYLOG logf
#define MYEXP expf

#endif

namespace irm { 

typedef boost::random::mt19937 rng_t;

typedef std::vector<int> axesdef_t; 
typedef std::vector<size_t> domainsizes_t; 
typedef int domainpos_t; 

typedef size_t dppos_t;
typedef size_t entitypos_t; 

typedef std::set<groupid_t> group_set_t; 



typedef fast_static_vector<entitypos_t, 2> entity_coords_t; 
const static int MAX_GROUPS_PER_DOMAIN_BITS = 8; 
const static int MAX_GROUPS_PER_DOMAIN = (1<<MAX_GROUPS_PER_DOMAIN_BITS); 

#define NOT_ASSIGNED -1


template<typename containerT, typename ForwardIterator> 
void cart_prod_helper(std::vector<containerT > & output, 
                          std::vector<std::pair<ForwardIterator, ForwardIterator> >  axes, 
                          containerT current_element, 
                          size_t axispos); 

template<typename containerT>
std::vector<containerT> cart_prod(std::vector<size_t> axes)
{
    std::vector<containerT> output; 
    containerT x(axes.size()); ; 
    // create the iterators
    typedef boost::counting_iterator<size_t> i_t; 
    std::vector<std::pair<i_t, i_t>> axes_iters; 

    for(auto a: axes) { 
        axes_iters.push_back(std::make_pair(i_t(0), i_t(a))); 
    }
    cart_prod_helper<containerT>(output, axes_iters, x, 0); 
    return output; 
}


// FIXME : use output iterator
// FIXME: use boost::range

template<typename containerT, typename ForwardIterator> 
void cart_prod_helper(std::vector<containerT > & output, 
                      std::vector<std::pair<ForwardIterator,
                      ForwardIterator>> axes, 
                      containerT current_element, 
                      size_t axispos) {
    for(auto it = axes[axispos].first; it != axes[axispos].second; ++it) {
        containerT x = current_element; 
        x[axispos] = *it; 
        
        if(axispos == (axes.size()-1)) { 
            output.push_back(x); 
        } else {
            cart_prod_helper(output, axes, x, axispos + 1); 
        }

    }
    
}



template<typename ForwardIterator> 
std::set<group_coords_t> unique_axes_pos(std::vector<int> axis_pos, size_t val, 
                                         std::vector<std::pair<ForwardIterator, ForwardIterator>> axes)

{
    /* 
       DIMS: maximum dimension of group coords

    */ 
    std::set<group_coords_t> outset; 
    std::vector<group_coords_t> output; 
    group_coords_t o(axes.size()); 

    cart_prod_helper<group_coords_t>(output, axes, o, 0); 

    for(auto c : output) { 
        bool include = false; 
        for(auto i: axis_pos) { 
            if (c[i] == (int)val) 
                include = true; 
        }
        if(include)
            outset.insert(c); 
    }
    return outset; 
    
}


template<typename T>
std::vector<std::pair<typename T::iterator, typename T::iterator>>
    collection_of_collection_to_iterators(std::vector<T> & collections) 
{
    std::vector<std::pair<typename T::const_iterator, 
                          typename T::const_iterator>>  output; 
    for(auto & a : collections) { 
        output.push_back(std::make_pair(a.begin(), a.end())); 
    }
    
    return output; 
}



inline float uniform(float min, float max, rng_t & rng) { 
 // boost::uniform_01<rng_t> dist(rng);  NEVER USE THIS IT DOES NOT ADVANCE THE RNG STATE

boost::random::uniform_real_distribution<> real(min, max); 
return real(rng);
}

inline float uniform_01(rng_t & rng) { 
 // boost::uniform_01<rng_t> dist(rng);  NEVER USE THIS IT DOES NOT ADVANCE THE RNG STATE
    // because uniform() above is [a, b], and we want [0, 1)
   return uniform(0, 0.999999f, rng); 
}

inline float log_exp_dist(float x, float lambda) {
    if(x <0.0) { 
            return -std::numeric_limits<float>::infinity();

    }
    if(lambda <=0.0) { 
            return -std::numeric_limits<float>::infinity();

    }
    return MYLOG(lambda) + -lambda*x; 

}

inline float log_norm_dist(float x, float mu, float sigmasq) {

    return -0.5 * MYLOG(sigmasq) - 0.5* MYLOG(2.0 * 3.14159265) + -(x - mu) * (x-mu) / (2.*sigmasq); 
    
}


inline float log_poisson_dist(int k, float lambda) {
    if(lambda <= 0.0) { 
            return -std::numeric_limits<float>::infinity();

    }
    float score = k * MYLOG(lambda) - lgammaf(k+1) + -lambda; 
    return score; 

}

inline float logbeta(float alpha, float beta) { 
    return fasterlgamma(alpha)  + fasterlgamma(beta) - fasterlgamma(alpha + beta); 
}

inline float log_factorial(int N) { 
    return lgammaf(N + 1.0); 
}

inline float log_beta_dist(float p, float alpha, float beta) {
    if((p <0.0) or (p > 1.0)) { 
            return -std::numeric_limits<float>::infinity();

    }
    float a = (alpha-1.)*MYLOG(p) + (beta-1.)*(MYLOG(1-p)); 
    float b = logbeta(alpha, beta); 
    
    return a + b; 

}

inline float normal_sample(float mean, float var, rng_t & rng) { 

    boost::random::normal_distribution<> real(mean, sqrt(var)); 
    return real(rng);
}

inline float chi2_sample(float v, rng_t & rng) { 
    boost::random::chi_squared_distribution<float> dist(v);
    return dist(rng); 

}
inline float beta_sample(float alpha, float beta, rng_t & rng) {

    boost::random::gamma_distribution<> g1(alpha, 1.0);
    boost::random::gamma_distribution<> g2(beta, 1.0);
        
    float g1val = g1(rng); 
    float g2val = g2(rng); 
    float c = g1val / (g1val + g2val);
    return c; 
}



inline float log_symmetric_dir_dist(const std::vector<float> pi, 
                                    float alpha)
{
    float score = 0.0; 
    int N = pi.size(); 
    for (int i = 0; i < pi.size(); ++i) { 
        score += (alpha - 1.0) * MYLOG(pi[i]); 
    }
    float beta = N * lgammaf(alpha) - lgammaf(alpha * N); 
    score -= beta; 
    return score ; 

}

inline std::vector<float> symmetric_dirichlet_sample(int N, float alpha, rng_t & rng) 
{
    //http://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
    std::vector<float> y(N); 
    boost::random::gamma_distribution<float> gamma(alpha);
    float sum = 0.0; 
    for (int i = 0; i < N; ++i) { 
        y[i] = gamma(rng); 
        sum += y[i]; 
    }
    for (int i = 0; i < N; ++i) { 
        y[i] = y[i] / sum; 
    }

    return y; 

}

inline float log_chi2_dist(float x, int k) { 

    if (x <= 0 ) { 
        return -std::numeric_limits<float>::infinity();
    }
    if (k <= 0 ) { 
        return -std::numeric_limits<float>::infinity();
    }
    float a = -(k/2.*MYLOG(2)  + lgammaf(k/2.0)); 
    float b = (k/2. - 1) * MYLOG(x) - x/2.0 ; 
    return a + b; 

}


inline float log_sum_exp(float x, float y) { 
    float a = 0; 
    float b = 0; 
    if (x > y) { 
        a = x;
        b = y; 
    } else { 
        a = y; 
        b = x; 
    }
    return a + MYLOG(1.0 + exp(b - a)); 
}

inline float log_t_pdf(float x, float nu, float mu, float sigmasq)
{
    /*
    Murphy, Eq. 304

    */
    float c = lgammaf(.5 * (nu + 1.))
        - (lgammaf(.5 * nu) + .5 * (MYLOG(nu * 3.1415926535f* sigmasq))); 
    float xt = (x - mu); 
    float s = xt * xt / sigmasq; 
    float d = -(.5 * (nu + 1.)) * MYLOG(1. + s / nu); 
    return c + d; 
    
}


}



#endif
 
