#include "pyirm_helpers.h"

bp::list cart_prod_helper_py(bp::list axes)
{
    std::vector<size_t> l; 
    
    for(int i = 0; i < bp::len(axes); i++) { 
        l.push_back(bp::extract<size_t>(axes[i])); 
    }
    std::vector<std::vector<size_t>> output = 
        irm::cart_prod<size_t>(l); 
    bp::list outl ; 
    for(auto i: output) { 
        bp::list e; 
        for(auto j : i) { 
            e.append(j); 
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
        for(auto v : c) { 
            coord.append(v); 
        }

        out.append(bp::tuple(coord)); 
    }
    return out; 
}
