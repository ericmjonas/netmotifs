#include "pyirm_helpers.h"
#include "kernels.h"

bp::list cart_prod_helper_py(bp::list axes)
{
    std::vector<size_t> l; 
    
    for(int i = 0; i < bp::len(axes); i++) { 
        l.push_back(bp::extract<size_t>(axes[i])); 
    }
    std::vector<std::vector<size_t>> output = 
        irm::cart_prod<std::vector<size_t>>(l); 
    bp::list outl ; 
    for(auto i: output) { 
        bp::list e; 
        for(int ji = 0; ji < len(axes); ++ji) { 
            e.append(i[ji]); 
        }
        outl.append(e); 
    }
    return outl; 


}

bp::list unique_axes_pos_helper_py(bp::list axes_pos, irm::groupid_t val, 
                                bp::tuple axes_possible_vals)
{

    using namespace irm; 

    std::vector<std::vector<groupid_t>> possible_vals; 
    std::vector<std::pair<std::vector<groupid_t>::iterator, 
                          std::vector<groupid_t>::iterator>> ranges; 
    for(int i = 0; i < bp::len(axes_possible_vals); ++i) { 
        bp::list ax_pv = bp::extract<bp::list>(axes_possible_vals[i]); 
        possible_vals.push_back(extract_vect<groupid_t>(ax_pv)); 
        auto r = std::make_pair(possible_vals.back().begin(), 
                                possible_vals.back().end()); 
        ranges.push_back(r); 
        
    }
    std::vector<int> axes_posv = extract_vect<int>(axes_pos); 
    std::set<group_coords_t> uap = irm::unique_axes_pos(axes_posv, val, 
                                                        ranges); 
    
    bp::list out; 
    for(auto c: uap ) { 
        bp::list coord; 
        for(int i = 0; i < len(axes_possible_vals); ++i){
            coord.append(c[i]); 
        }

        out.append(bp::tuple(coord)); 
    }
    return out; 
}

float slice_sampler_wrapper(float x, bp::object P, 
                            irm::rng_t & rng, float w) { 
    /* 
       Only works for floats, oh well
    */ 

    return irm::slice_sample<float>(x,
                                    [&P](float t) -> float
        { return bp::extract<float>(P(t));}, 
                                    rng, w);


}


float continuous_mh_sampler_wrapper(float x, bp::object P, 
                                    irm::rng_t & rng, int iters, 
                                    float scale_min, float scale_max) { 
    /* 
       Only works for floats, oh well
    */ 

    return irm::continuous_mh_sample<float>(x,
                                    [&P](float t) -> float
        { return bp::extract<float>(P(t));}, 
                                            rng, iters, scale_min, scale_max);


}
