#ifndef __IRM_PYIRM_HELPERS_H__
#define __IRM_PYIRM_HELPERS_H__
#include <unistd.h>

#include <boost/python.hpp>
#include <boost/python/class.hpp>

#include "util.h"
#include "relation.h"
#include "componentcontainer.h"

namespace bp=boost::python;

template<typename T, typename bpT> 
std::vector<T> extract_vect(bpT l) { 
    std::vector<T> out; 
    for(int i =0; i < bp::len(l); ++i) { 
        out.push_back(bp::extract<T>(l[i])); 
    }
    return out; 
}

bp::list cart_prod_helper_py(bp::list axes); 

bp::list unique_axes_pos_helper_py(bp::list axes_pos, irm::groupid_t val, 
                                bp::tuple axes_possible_vals); 

float slice_sampler_wrapper(float x, bp::object P, 
                            irm::rng_t & rng, float w); 

float continuous_mh_sampler_wrapper(float x, bp::object P, 
                                    irm::rng_t & rng, int iters, 
                                    float scale_min, float scale_max); 

#endif
