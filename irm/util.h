#ifndef __IRM_UTIL_H__
#define __IRM_UTIL_H__

#include <list>
#include <set>
#include <vector>
#include <stdlib.h>

namespace irm { 

const int MAX_AXES = 10; 

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

}


#endif
