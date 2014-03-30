#ifndef __IRM_PARRELATION_H__
#define __IRM_PARRELATION_H__

#include <boost/timer/timer.hpp>

#include "util.h"
#include "componentcontainer.h"

namespace irm { 

class ParRelation
{
/*
  Domains are numbered, and axes are a list of domain integers. So for a T1xT1 
  we would specify 
  
  
*/
public:
    ParRelation(axesdef_t axes_def, domainsizes_t domainsizes, 
             IComponentContainer * cm); 
    ~ParRelation() ; 
    void assert_unassigned(); 
    void assert_assigned();
    size_t assigned_dp_count();

    groupid_t create_group(domainpos_t domain, rng_t & rng); 
    /* Creates groups and initializes from the prior, thus the need
    for the RNG */ 
    
    void delete_group(domainpos_t dom, groupid_t gid); 

    std::vector<groupid_t> get_all_groups(domainpos_t); 

    float add_entity_to_group(domainpos_t, groupid_t, entitypos_t); 
    
    void remove_entity_from_group(domainpos_t, groupid_t, entitypos_t); 

    float post_pred(domainpos_t, groupid_t, entitypos_t); 

    float post_pred_combined(domainpos_t, groupid_t, entitypos_t) const; 

    // postpred across all groups
    bp::list post_pred_map(domainpos_t, 
                           bp::list groups, 
                           entitypos_t); 

    group_dp_map_t get_datapoints_per_group(); 

    float total_score(); 
    bp::dict get_component(bp::tuple group_coords); 
    void set_component(bp::tuple group_coords, bp::dict params); 

private:
    const axesdef_t axes_; 
    IComponentContainer * pCC_; 
    const int DIMS_; 
    const int DOMAINN_; 
    domainsizes_t domainsizes_; 
    std::vector<std::vector<int> > domain_to_axispos_; 

    std::vector<std::vector<groupid_t>> group_ids_; 

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
    set_dp_group_coords(dppos_t dp, const group_coords_t &  gc) { 
        datapoints_per_group_cache_valid_ = false; 

         datapoint_groups_[dp] = gc; 
    }
    
    inline groupid_t get_entity_group(domainpos_t domain, entitypos_t ep) const { 
        return domain_entity_assignment_[domain][ep]; 
    }
    
    inline void set_entity_group(domainpos_t domain, entitypos_t ep, groupid_t g) { 
        datapoints_per_group_cache_valid_ = false; 

        domain_entity_assignment_[domain][ep] = g; 
    }

    inline bool fully_assigned(const group_coords_t & gc) const { 
        for (int i = 0; i < DIMS_; ++i) { 
            if (gc[i] == NOT_ASSIGNED) 
                return false; 
        }
        return true; 

    }
    bool datapoints_per_group_cache_valid_; 
    
    group_dp_map_t datapoints_per_group_cache_; 

    

}; 

}


#endif 
