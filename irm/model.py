import numpy as np
import util
import gibbs

"""
Operations we want to be fast: 

Adding suffstats across a relation
Removing suffstats across a relation
computing post pred across a relation

We need to handle the T1xT1 case, which is the most frustrating case

Data must be sparse
A relation must have hypers


"""
NOT_ASSIGNED = -1


class DomainInterface(object):
    """
    A single handle that we use to glue objects together
    Also computes the CRP
    """

    def __init__(self, ENT_N, relations_dict, fixed_k = False):
        """
        relations : {'relationname' : ('DOMAINNAME', rel_obj)}
        where domainname is the name the relation knows this domain as


        """
        self.relation_pos = {}
        relation_list = []
        for ri, (relation_name, relation_def) in enumerate(relations_dict.iteritems()):
            self.relation_pos[relation_name] = ri
            relation_list.append(relation_def)

        self.relations = relation_list
        self.gid_mapping = {}
        self.g_pos = 0
        self.assignments = np.ones(ENT_N, dtype=np.int)
        self.assignments[:] = NOT_ASSIGNED
        self.temp = 1.0

        self.fixed_k = fixed_k
        print "This model fixed_k= ", self.fixed_k

    def get_relation_pos(self, relation_name):
        return self.relation_pos[relation_name]

    def entity_count(self):
        return len(self.assignments)

    def set_hps(self, hps):
        self.alpha = hps['alpha']
    
    def get_hps(self):
        return {'alpha' : self.alpha}

    def get_groups(self):
        return self.gid_mapping.keys()

    def group_count(self):
        return len(self.gid_mapping)

    def set_temp(self, t):
        self.temp = t

    def create_group(self, rng):
        """
        Returns group ID
        """
        rel_groupid = [r.create_group(t, rng) for t, r in self.relations]
        
        new_gid = self.g_pos
        self.g_pos += 1
        self.gid_mapping[new_gid] = tuple(rel_groupid)

        #if len(self.gid_mapping) > self.fixed_k:
        #    print "WARNING: Creating too many groups in fixed K configuration" 

        return new_gid
    
    def group_size(self, gid):
        """
        How many entities in this group
        """
        #FIXME slow
        return np.sum(self.assignments == gid)

    def _assigned_entity_count(self):
        return np.sum(self.assignments != NOT_ASSIGNED)

    def get_assignments(self):
        return self.assignments.copy()

    def get_relation_groupid(self, relpos, g):
        """
        Get the underlying relation's groupid for this group
        
        Useful for then getting at relation components
        """
        return self.gid_mapping[g][relpos]

    def delete_group(self, group_id):
        rel_groupid = self.gid_mapping[group_id]
        [r.delete_group(t, g) for g, (t, r) in zip(rel_groupid, self.relations)]
        
        del self.gid_mapping[group_id]

        #if self.fixed_k:
        #    print "WARNING: Deleting group in Fixed K configuration"

    def add_entity_to_group(self, group_id, entity_pos):
        assert self.assignments[entity_pos] == NOT_ASSIGNED
        rel_groupid = self.gid_mapping[group_id]
        [r.add_entity_to_group(t, g, entity_pos) for g, (t, r) in zip(rel_groupid, 
                                                            self.relations)]
        self.assignments[entity_pos] = group_id

    def remove_entity_from_group(self, entity_pos):
        assert self.assignments[entity_pos] != NOT_ASSIGNED
        group_id = self.assignments[entity_pos]
        rel_groupid = self.gid_mapping[group_id]
        [r.remove_entity_from_group(t, g, entity_pos) for g, (t, r) \
        in zip(rel_groupid, 
            self.relations)]

        self.assignments[entity_pos] = NOT_ASSIGNED
        return group_id

    def get_prior_score(self):
        count_vect = util.assign_to_counts(self.assignments)
        if not self.fixed_k :
            score = util.crp_score(count_vect, self.alpha)
        else:
            score = util.dm_score(count_vect, self.alpha, self.group_count())
        return score
        
    def post_pred(self, group_id, entity_pos):
        """
        Combines likelihood and the CRP
        """
        # can't post-pred an assigned row
        assert self.assignments[entity_pos] == NOT_ASSIGNED
        
        rel_groupid = self.gid_mapping[group_id]
        scores = [r.post_pred(t, g, entity_pos) for g, (t, r) in zip(rel_groupid, 
                                                                    self.relations)]
        gc = self.group_size(group_id)
        assigned_entity_N = self._assigned_entity_count()

        if not self.fixed_k:
            prior_score = util.crp_post_pred(gc, assigned_entity_N+1, self.alpha)
        else:
            prior_score = util.dm_post_pred(gc, assigned_entity_N+1, self.alpha, 
                                            self.group_count())
            
        return np.sum(scores) + prior_score/self.temp

    def post_pred_map(self, groups, entity_pos, threadpool = None):
        """
        Combines likelihood and the CRP
        """
        # can't post-pred an assigned row
        assert self.assignments[entity_pos] == NOT_ASSIGNED

        assigned_entity_N = self._assigned_entity_count()

        
        scores = np.zeros(len(groups))
        rel_group_ids = np.zeros((len(self.relations), len(groups)), 
                                 dtype=np.int)
        sizes = np.zeros(len(groups))
        for gi, g in enumerate(groups):
            rel_group_ids[:, gi] = self.gid_mapping[g]

            gc = self.group_size(g)

            if not self.fixed_k:
                prior_score = util.crp_post_pred(gc, assigned_entity_N+1, self.alpha)
            else:
                prior_score = util.dm_post_pred(gc, assigned_entity_N+1, self.alpha, 
                                                self.group_count())
            scores[gi] += prior_score / self.temp

        for ri, (t, r) in enumerate(self.relations):
            scores += r.post_pred_map(t, rel_group_ids[ri].tolist(),
                                      entity_pos, threadpool)
            
        return scores

class IRM(object):
    def __init__(self, domains, relations):
        """
        both domains and relations are name->object mappings
        """
        self.domains = domains
        self.relations = relations
    
    def set_temp(self, t):
        for r in self.relations.values():
            r.set_temp(t)

        for d in self.domains.values():
            d.set_temp(t)

    def get_score(self):
        return self.total_score()

    def total_score(self):
        score = 0
        for t in self.domains.values():
            score += t.get_prior_score()
        for r in self.relations.values():
            score += r.total_score()
        return score


def get_components_in_relation(domains_as_axes, relation):
    """
    Return the components for a given relation indexed by
    the axes tuples of the outer (python) domains

    domains_as_axes: [(domain_obj, relpos)] 
    
    """
    
    possible_group_axes = [d[0].get_groups() for d in domains_as_axes]
    domain_coords = util.cart_prod(possible_group_axes)

    comp_vals = {}

    for ci, coord in enumerate(domain_coords):
        rel_coords = []
        for di, (dobj, relpos) in enumerate(domains_as_axes):
            r_gid = dobj.get_relation_groupid(relpos, coord[di])
            rel_coords.append(r_gid)
        c = relation.get_component(tuple(rel_coords))
        comp_vals[coord] = c
    return comp_vals

def set_components_in_relation(domains_as_axes, relation, 
                               coord, val):
    """
    Set the component in the relation, nbut using the domain's
    GIDs -- perform the mapping

    domains_as_axis = [(domain_object, what is the position in the 
    domain for this relation)]

    """
    if not relation.conjugate:
        rel_coords = []
        for di, (dobj, relpos) in enumerate(domains_as_axes):
            r_gid = dobj.get_relation_groupid(relpos, coord[di])
            rel_coords.append(r_gid)

        relation.set_component(tuple(rel_coords), val)
    
