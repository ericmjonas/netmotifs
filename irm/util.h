#ifndef __IRM_UTIL_H__
#define __IRM_UTIL_H__

#include <list>
#include <set>
#include <vector>
#include <array>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <stdlib.h>

#define GROUP_COORDS_ARRAY
#include "group_coords.h"


namespace irm { 

typedef boost::random::mt19937 rng_t;

typedef std::vector<int> axesdef_t; 
typedef std::vector<size_t> domainsizes_t; 
typedef int domainpos_t; 

typedef size_t dppos_t;
typedef size_t entitypos_t; 

typedef std::set<groupid_t> group_set_t; 



typedef std::vector<entitypos_t> entity_coords_t; 

const static int MAX_GROUPS_PER_DOMAIN = 1000; 
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
    group_coords_t o = new_group_coords(axes.size()); 

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
   return uniform(0, 1.0-1e-9, rng); 
}

inline float log_exp_dist(float x, float lambda) {
    if(x <0.0) { 
            return -std::numeric_limits<float>::infinity();

    }
    return logf(lambda) + -lambda*x; 

}

}



#endif
