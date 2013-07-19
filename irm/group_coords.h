#ifndef __IRM_GROUP_COORDS_H__
#define __IRM_GROUP_COORDS_H__

#include <vector>
#include <array>
#include "fast_static_vect.h"

namespace irm { 

typedef int16_t groupid_t; 
static const int MAX_AXES = 2; 


typedef fast_static_vector<groupid_t, MAX_AXES> group_coords_t; 

// inline group_coords_t new_group_coords(int size) 
// {
//     return group_coords_t(size); 
    
// }

// inline group_coords_t new_group_coords(int size, groupid_t defval) 
// {
//     return group_coords_t(size, defval); 

// }

}


#endif
