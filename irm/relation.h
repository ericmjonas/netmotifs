#ifndef __IRM_RELATION_H__
#define __IRM_RELATION_H__

#include "util.h"
#include "componentcontainer.h"

namespace irm { 

class Relation
{
    /*
      Domains are numbered, and axes are a list of domain integers. So for a T1xT1 
      we would specify 
      
      
    */
public:
    Relation(axesdef_t axes_def, domainsizes_t domainsizes, 
             IComponentContainer * cm); 
    
    void assert_unassigned(); 
    void assert_assigned();
    size_t assigned_dp_count();

    groupid_t create_group(domainpos_t domain); 
    
    void delete_group(domainpos_t dom, groupid_t gid); 

    std::vector<groupid_t> get_all_groups(domainpos_t); 

    float add_entity_to_group(domainpos_t, groupid_t, entitypos_t); 
    
    float remove_entity_from_group(domainpos_t, groupid_t, entitypos_t); 
    float post_pred(domainpos_t, groupid_t, entitypos_t); 

    float total_score(); 
private:
    const axesdef_t axes_; 
    IComponentContainer * pCC_; 
    const int DIMS_; 
    const int DOMAINN_; 
    std::vector<std::vector<int> > domain_to_axispos_; 

    groupid_t group_id_; 

    // maps from a datapoint to that datapoint's current group coordinates
    std::vector<group_coords_t> datapoint_groups_; 
    // maps from dp index to entity coordinates
    std::vector<entity_coords_t> datapoint_entity_index_; 

    // for each domain, maps from its entities to the domain
    std::vector<std::vector<std::vector<int> > > entity_to_dp; 
    
}; 

}


#endif 
