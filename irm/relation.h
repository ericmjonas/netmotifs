#ifndef __IRM_RELATION_H__
#define __IRM_RELATION_H__

#include <boost/timer/timer.hpp>

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
    ~Relation() ; 
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
    domainsizes_t domainsizes_; 
    std::vector<std::vector<int> > domain_to_axispos_; 

    std::vector<groupid_t> group_ids_; 

    // maps from a datapoint to that datapoint's current group coordinates
    std::vector<group_coords_t> datapoint_groups_; 
    // maps from dp index to entity coordinates
    std::vector<entity_coords_t> datapoint_entity_index_; 

    // for each domain, maps from its entities to the domain
    std::vector<std::vector<std::vector<dppos_t> > > entity_to_dp_; 

    // for each domain, map entitypos to groupid
    std::vector<std::vector<groupid_t>> domain_entity_assignment_; 

    // map from domain to the groups contained in that domain 
    std::vector<group_set_t> domain_groups_; 

    // private functions
    inline const std::vector<int> & 
    get_axispos_for_domain(domainpos_t dp) const { 
        return domain_to_axispos_[dp]; 
    }
    
    inline const std::vector<dppos_t> & 
    datapoints_for_entity(domainpos_t domain, entitypos_t ep) const {
        return entity_to_dp_[domain][ep]; 
    }
    
    inline const entity_coords_t &
    get_dp_entity_coords(dppos_t dp) const { 
        return datapoint_entity_index_[dp]; 
    }
    
    inline const group_coords_t & 
    get_dp_group_coords(dppos_t dp) const { 
        return datapoint_groups_[dp]; 
    }

    inline void
    set_dp_group_coords(dppos_t dp, group_coords_t gc) { 
         datapoint_groups_[dp] = gc; 
    }
    
    inline groupid_t get_entity_group(domainpos_t domain, entitypos_t ep) const { 
        return domain_entity_assignment_[domain][ep]; 
    }
    
    inline void set_entity_group(domainpos_t domain, entitypos_t ep, groupid_t g) { 
        domain_entity_assignment_[domain][ep] = g; 
    }

    inline bool fully_assigned(group_coords_t gc) { 
        for(auto g : gc) { 
            if (g == NOT_ASSIGNED) 
                return false; 
        }
        return true; 

    }
 
    
    

}; 

}


#endif 
