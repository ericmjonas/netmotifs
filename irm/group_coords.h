#ifndef __IRM_GROUP_COORDS_H__
#define __IRM_GROUP_COORDS_H__

#include <vector>
#include <array>
#include "fast_static_vect.h"

namespace irm { 

typedef int16_t groupid_t; 
static const int MAX_AXES = 2; 

#ifdef GROUP_COORDS_ARRAY

typedef std::array<groupid_t, MAX_AXES> group_coords_t; 

inline group_coords_t new_group_coords(int size) 
{
    group_coords_t c; 
    for(int i = 0; i < MAX_AXES; ++i) { 
        c[i] = 0; 
    }
    return c; 
    
}

inline group_coords_t new_group_coords(int size, groupid_t defval) 
{
    group_coords_t c; 
    for(int i = 0; i < size; ++i) { 
        c[i] = defval; 
    }
    return c; 
}
#endif

#ifdef GROUP_COORDS_VECT

typedef std::vector<groupid_t> group_coords_t; 
inline group_coords_t new_group_coords(int size) 
{
    return group_coords_t(size); 
    
}

inline group_coords_t new_group_coords(int size, groupid_t defval) 
{
    return group_coords_t(size, defval); 

}
#endif


#ifdef GROUP_COORDS_FAST_STATIC_VECTOR

typedef fast_static_vector<groupid_t, MAX_AXES> group_coords_t; 

inline group_coords_t new_group_coords(int size) 
{
    return group_coords_t(size); 
    
}

inline group_coords_t new_group_coords(int size, groupid_t defval) 
{
    return group_coords_t(size, defval); 

}
#endif


}


#endif
