#include <boost/assign.hpp>

#include "relation.h"


namespace irm { 

Relation::Relation(axesdef_t axes_def, domainsizes_t domainsizes, 
                   IComponentContainer * cm) :
    axes_(axes_def), 
    pCC_(cm), 
    DIMS_(axes_def.size()), 
    DOMAINN_(domainsizes.size()), 
    domainsizes_(domainsizes), 
    group_ids_(DOMAINN_), 
    datapoint_groups_(pCC_->dpcount()), 
    datapoint_entity_index_(pCC_->dpcount()), 
    domain_groups_(DOMAINN_)
{

    for (int d = 0; d < DOMAINN_; ++d) { 
        std::vector<int> axispos; 
        for (size_t i = 0; i < axes_.size(); ++i) { 
            if(axes_[i] == d) { 
                axispos.push_back(i); 
            }
            
        }
        domain_to_axispos_.push_back(axispos); 
        domain_entity_assignment_.push_back( std::vector<groupid_t>(domainsizes_[d], NOT_ASSIGNED)); 

        entity_to_dp_.push_back( std::vector<std::vector<dppos_t>>(domainsizes_[d])); 

    }
    
    for(size_t dpi = 0; dpi < pCC_->dpcount(); ++dpi) { 
        datapoint_groups_[dpi] = group_coords_t(DIMS_, NOT_ASSIGNED); 
    }
    
    // do enttiy_to_dp
    
    std::vector<size_t> axes_sizes; 
    for(int i = 0; i < DIMS_; i++) { 
        axes_sizes.push_back(domainsizes_[axes_[i]]); 
    }

    dppos_t dppos = 0; 
    for(auto dp_coord : cart_prod<entity_coords_t >(axes_sizes)) { 
        datapoint_entity_index_[dppos] = dp_coord; 
        
        for(int ai = 0; ai < DIMS_; ai++) { 
            auto eid = dp_coord[ai]; 
            domainpos_t di = axes_[ai]; 
            auto & e_dp = entity_to_dp_[di][eid]; 
            if(std::find(e_dp.begin(), e_dp.end(), dppos) == e_dp.end()) { 
                entity_to_dp_[di][eid].push_back(dppos); 
            }
        }
        assert(dppos < pCC_->dpcount()); 
        dppos++; 
    }

    // initial stacks of group ids
    for(int i = (MAX_GROUPS_PER_DOMAIN-1); i >= 0; i--) { 
        for(int d = 0; d < DOMAINN_; ++d) { 
            group_ids_[d].push_back(i); 
        }
    }


}


void Relation::assert_unassigned() { 


}


void Relation::assert_assigned() {
    

}

size_t Relation::assigned_dp_count()
{
    // FIXME: implement
    return 0; 
}


groupid_t Relation::create_group(domainpos_t domain, rng_t & rng)
{
    if(group_ids_[domain].empty()) { 
        throw std::runtime_error("too many groups!"); 
    }
    groupid_t new_gid = group_ids_[domain].back(); 
    domain_groups_[domain].insert(new_gid); 
    // domains by 
    std::vector<group_set_t > domains_as_axes; 
    for(auto a : axes_) { 
        domains_as_axes.push_back(domain_groups_[a]); 
    }

    auto domain_iters = collection_of_collection_to_iterators(domains_as_axes);
    auto group_coords = unique_axes_pos(get_axispos_for_domain(domain), 
                                        new_gid, domain_iters); 

    for(auto g : group_coords) { 
        pCC_->create_component(g, rng); 
    }

    group_ids_[domain].pop_back(); 

    return new_gid; 
}

void Relation::delete_group(domainpos_t domain, groupid_t gid)
{
    // FIXME: god this is inefficient
    std::vector<group_set_t > domains_as_axes; 
    for(auto a : axes_) { 
        domains_as_axes.push_back(domain_groups_[a]); 
    }

    auto domain_iters = collection_of_collection_to_iterators(domains_as_axes);

    auto group_coords = unique_axes_pos(get_axispos_for_domain(domain), 
                                        gid,
                                        domain_iters); 
    
    for(auto g : group_coords) { 
        pCC_->delete_component(g); 
    }
    domain_groups_[domain].erase(gid); 

    // put it back on the stack
    group_ids_[domain].push_back(gid); 
}

std::vector<groupid_t> Relation::get_all_groups(domainpos_t di)
{
    return std::vector<groupid_t>(domain_groups_[di].begin(), 
                                  domain_groups_[di].end());
}

// FIXME come up with some way of caching the mutated components
float Relation::add_entity_to_group(domainpos_t domain, groupid_t group_id, 
                                    entitypos_t entity_pos)
{

    const auto & axispos_for_domain = get_axispos_for_domain(domain); 

    float score = 0.0; 

    for(auto dp :  datapoints_for_entity(domain, entity_pos)) { 
        const auto & current_group_coords = get_dp_group_coords(dp); 
        auto new_group_coords = current_group_coords; 
        const auto & dp_entity_pos = get_dp_entity_coords(dp); 
        
        for(auto axis_pos :  axispos_for_domain) { 
            if(dp_entity_pos[axis_pos] == entity_pos) { 
                new_group_coords[axis_pos] = group_id; 
            }
        }
        if (fully_assigned(new_group_coords) and !fully_assigned(current_group_coords))
            {
                score += pCC_->add_dp_post_pred(new_group_coords, dp); 
            }

        set_dp_group_coords(dp, new_group_coords); 
    }

    set_entity_group(domain, entity_pos, group_id); 
    return score; 
}

void Relation::remove_entity_from_group(domainpos_t domain, groupid_t groupid, 
                                        entitypos_t entity_pos)
{

    const auto & axispos_for_domain = get_axispos_for_domain(domain); 
    for(auto dp : datapoints_for_entity(domain, entity_pos)) { 
        const auto & current_group_coords = get_dp_group_coords(dp); 
        auto new_group_coords = current_group_coords; 
        const auto & dp_entity_pos = get_dp_entity_coords(dp); 
        
        for(auto axis_pos : axispos_for_domain) { 
            if(dp_entity_pos[axis_pos] == entity_pos) { 
                new_group_coords[axis_pos] = NOT_ASSIGNED; 
            }
        }
        if (fully_assigned(current_group_coords)) { 
            pCC_->rem_dp(current_group_coords, dp); 
        }
        set_dp_group_coords(dp, new_group_coords); 
    }
    set_entity_group(domain, entity_pos, NOT_ASSIGNED); 
}


float Relation::post_pred(domainpos_t domain, groupid_t group_id, 
                          entitypos_t entity_pos)
{
    float score = add_entity_to_group(domain, group_id, entity_pos); 
    remove_entity_from_group(domain, group_id, entity_pos); 
    return score; 


}

float Relation::total_score()
{

    return pCC_->total_score(get_datapoints_per_group()); 

}

Relation::~Relation()
{

}

bp::dict Relation::get_component(bp::tuple group_coords)
{
    group_coords_t gc(DIMS_); 
    for(int i = 0; i < bp::len(group_coords); ++i) { 
        gc[i] = bp::extract<int>(group_coords[i]); 
    }
    return pCC_->get_component(gc); 
}


void Relation::set_component(bp::tuple group_coords, 
                                 bp::dict val)
{
    group_coords_t gc(DIMS_); 
    for(int i = 0; i < bp::len(group_coords); ++i) { 
        gc[i] = bp::extract<int>(group_coords[i]); 
    }
    return pCC_->set_component(gc, val); 
}

group_dp_map_t Relation::get_datapoints_per_group()
{
    group_dp_map_t out; 
    
    for(size_t i = 0; i < datapoint_groups_.size(); ++i) { 
        
        group_coords_t gc = get_dp_group_coords(i); 

        if (fully_assigned(gc)) { 
            if(out.find(gc) == out.end()){ 
                out.insert(std::make_pair(gc, std::vector<dppos_t>())); 
            }
            out[gc].push_back(i); 
        }
    }
    return out; 

}


}
