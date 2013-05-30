#ifndef __IRM_UTIL_H__
#define __IRM_UTIL_H__

#include <list>
#include <set>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>
#include <stdlib.h>

namespace irm { 

const int MAX_AXES = 4; 

typedef std::vector<int> axesdef_t; 
typedef std::vector<size_t> domainsizes_t; 
typedef int domainpos_t; 
typedef int groupid_t; 
typedef size_t dppos_t;
typedef size_t entitypos_t; 

typedef std::set<groupid_t> group_set_t; 

typedef std::vector<groupid_t> group_coords_t; 
typedef std::vector<entitypos_t> entity_coords_t; 


#define NOT_ASSIGNED -1

// template<typename T> 
// void cart_prod_helper(std::vector<std::vector<T> > & output, 
//                       std::vector<size_t> axes, 
//                       std::vector<T> current_element, 
//                       int axispos); 

// template<typename T> 
// void cart_prod_helper(std::vector<std::vector<T> > & output, 
//        std::vector<size_t> axes, 
//        std::vector<T> current_element, 
//        int axispos) {
//     if(axispos == (axes.size()-1)) { 
//         // base case
//         for(int i = 0; i < axes[axispos]; ++i) {
//             std::vector<T> x = current_element; 
//             x.push_back(i); 
//             output.push_back(x); 
//         }
//     } else { 
//         for(int i = 0; i < axes[axispos]; ++i) { 
//             std::vector<T> x = current_element; 
//             x.push_back(i); 
//             cart_prod_helper(output, axes, x, axispos + 1); 
//         }
//     }

    
// }




template<typename T, typename ForwardIterator> 
void cart_prod_helper(std::vector<std::vector<T> > & output, 
                      std::vector<std::pair<ForwardIterator, ForwardIterator> >  axes, 
                      std::vector<T> current_element, 
                      int axispos); 

template<typename T>
std::vector<std::vector<T>> cart_prod(std::vector<size_t> axes)
{
    std::vector<std::vector<T>> output; 
    std::vector<T> x; 
    // create the iterators
    typedef boost::counting_iterator<size_t> i_t; 
    std::vector<std::pair<i_t, i_t>> axes_iters; 

    for(auto a: axes) { 
        axes_iters.push_back(std::make_pair(i_t(0), i_t(a))); 
    }
    cart_prod_helper<T>(output, axes_iters, x, 0); 
    return output; 
}


// FIXME : use output iterator
// FIXME: use boost::range

template<typename T, typename ForwardIterator> 
void cart_prod_helper(std::vector<std::vector<T> > & output, 
                      std::vector<std::pair<ForwardIterator, ForwardIterator>> axes, 
       std::vector<T> current_element, 
       int axispos) {
    for(auto it = axes[axispos].first; it != axes[axispos].second; ++it) {
        if(axispos == (axes.size()-1)) { 
            std::vector<T> x = current_element; 
            x.push_back(*it); 
            output.push_back(x); 
        } else {
            std::vector<T> x = current_element; 
            x.push_back(*it); 
            cart_prod_helper(output, axes, x, axispos + 1); 
        }
    }
    
}

}



#endif
