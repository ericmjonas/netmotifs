#include "relation.h"

namespace irm { 

Relation::Relation(axesdef_t axes_def, domainsizes_t domainsizes, 
                   IComponentContainer * cm) :
    axes_(axes_def), 
    pCC_(cm), 
    DIMS_(axes_def.size()), 
    DOMAINN_(domainsizes.size()), 
    group_id_(0), 
    datapoint_groups_(pCC_->dpcount()), 
    datapoint_entity_index_(pCC_->dpcount())
{
    // for (int d = 0; d < DOMAINN_; ++d) { 
    //     std::vector<int> axispos; 
    //     for (int i = 0; i < axes_.size(); ++i) { 
    //         if(axes_[i] == d) { 
    //             axispos.push_back(d); 
    //         }
            
    //     }
    //     domain_to_axispos_.push_back(axispos); 
    //     domain_entity_assignment_[d] = boost::assign::repeat(domainsizes_[d], NOT_ASSIGNED); 
        
    //     domain_groups[d].push_back(group_set_t()); 
    // }
       
    
    // for(int dpi = 0; dpi < pCC_->dpcount(); ++dpi) { 
    //     datapoint_groups[dpi] = boost::assign::repeat(DIMS_, NOT_ASSIGNED)
    // }

    // FIXME: add in the cartesian product coordinate stuff

}


void Relation::assert_unassigned() { 


}


void Relation::assert_assigned() {
    

}

size_t Relation::assigned_dp_count()
{
    // FIXME: implement

}

std::vector<int> get_axispos_for_domain(domainpos_t domain) 
{


}

std::vector<int> datapoints_for_entity(domainpos_t domain, entitypos_t entitypos)
{


}

std::vector<int> get_dp_entity_coords(dppos_t dp)
{

}


groupid_t Relation::create_group(domainpos_t domain)
{

}

void Relation::delete_group(domainpos_t dom, groupid_t gid)
{


}

std::vector<groupid_t> Relation::get_all_groups(domainpos_t)
{


}

// FIXME come up with some way of caching the mutated components
float Relation::add_entity_to_group(domainpos_t domain, groupid_t group_id, 
                                    entitypos_t entity_pos)
{

    // auto axispos_for_domain = get_axispos_for_domain(domain); 

    // float score = 0.0; 

    // BOOST_FOREACH(auto dp, datapoints_for_entity(domain, entity_pos)) { 
    //     auto current_group_coords = get_dp_group_coords(dp); 
    //     auto new_group_coords = copy(current_group_coords); 
    //     auto dp_entity_pos = get_dp_entity_coords(dp); 
        
    //     BOOST_FOREACH(auto axis_pos in axispos_for_domain) { 
    //         if(dp_entity_pos[axis_pos] == entity_pos) { 
    //             new_group_coords[axis_pos] = group_id; 
    //         }
    //     }
    //     if (fully_assigned(new_group_coords) and !fully_assigned(current_group_coords))
    //         {
    //             score += cm->post_pred(new_group_coords, dp)
    //             cm->add_dp(new_group_coords, dp); 
    //         }

    //     set_dp_group_coords(dp, new_group_coords); 
    // }

    // set_entity_group(domain, entity_pos, group_id); 
    // return score; 
}

float Relation::remove_entity_from_group(domainpos_t domain, groupid_t groupid, 
                                        entitypos_t entity_pos)
{

    // float score = 0.0; 
    // auto axispos_for_domain = get_axispos_for_domain(domain); 
    // BOOST_FOREACH(auto dp, datapoints_for_entity(domain, entity_pos)) { 
    //     auto current_group_coords = get_dp_group_coords(dp); 
    //     auto new_group_coords = copy(current_group_coords); 
    //     auto dp_entity_pos = get_dp_entity_coords(dp); 
        
    //     BOOST_FOREACH(auto axis_pos, axispos_for_domain) { 
    //         if(dp_entity_pos[axis_pos] == entity_pos) { 
    //             new_group_coords[axis_pos] = NOT_ASSIGNED; 
    //         }
    //     }
    //     if (fully_assigned(current_group_coordinates)) { 
    //         cm->rem_dp(group_coordinates, dp); 
    //         score -= post_pred(group_coordinates, dp); 
    //     }
    //     set_dp_group_coords(dp, new_group_coords); 
    // }
    // set_entity_group(domain, entity_pos, NOT_ASSIGNED); 
    // return score; 
}


float Relation::post_pred(domainpos_t domain, groupid_t groupid, 
                          entitypos_t entitypos)
{
    // float score = add_entity_to_group(domain, groupid, entitypos); 
    // remove_entity_from_group(domain, groupid, entitypos); 
    // return score; 
}

float Relation::total_score()
{

    return pCC_->total_score(); 

}

}
