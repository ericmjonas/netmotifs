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

    def __init__(self, ENT_N, relations):
        """
        relations : ('TYPENAME', rel_obj)
        """
        self.relations = relations
        self.gid_mapping = {}
        self.g_pos = 0
        self.assignments = np.ones(ENT_N, dtype=np.int)
        self.assignments[:] = NOT_ASSIGNED

    def entity_count(self):
        return len(self.assignments)

    def set_hps(self, alpha):
        self.alpha = alpha

    def get_groups(self):
        return self.gid_mapping.keys()

    def create_group(self, rng):
        """
        Returns group ID
        """
        rel_groupid = [r.create_group(t, rng) for t, r in self.relations]
        
        new_gid = self.g_pos
        self.g_pos += 1
        self.gid_mapping[new_gid] = tuple(rel_groupid)
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
        crp_score = util.crp_score(count_vect, self.alpha)
        return crp_score
        
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
        
        prior_score = util.crp_post_pred(gc, assigned_entity_N+1, self.alpha)
        
        return np.sum(scores) + prior_score

class IRM(object):
    def __init__(self, types, relations):
        """
        both types and relations are name->object mappings
        """
        self.types = types
        self.relations = relations
    
    def total_score(self):
        score = 0
        for t in self.types.values():
            score += t.get_prior_score()
        for r in self.relations.values():
            score += r.total_score()
        return score

