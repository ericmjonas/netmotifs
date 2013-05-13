import numpy as np
import util


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

class Relation(object):
    """
    Low-level implementaiton of a relation. 

    Note that, 
    because types can be shared across relations, we do NOT include
    the CRP/PY prior in these calculations. 

    DO NOT rely on this module to keep track of canonical object assignments
    
    """
    
    def __init__(self, type_def, data, modeltype = None):
        """
        type_def: [('NAME', ENT_N), ('NAME', ENT_N)]
        Data is anything where we can index-lookup and get

        """

        assert len(type_def) > 1
        
        self.relation_def = type_def
        
        self.data = data
        self.type_names = set()
        self.type_sizes = {}
        
        # type groups : the canoncial list of groups for a given type
        self.type_groups = {}

        self.group_pos = {}
        self.group_assignments = {}
        for name, ent_n in type_def:
            if name in self.type_names:
                # check if params are the same
                assert self.type_sizes[name] == ent_n
                assert self.type_groups[name] == set()
            else:
                self.type_names.add(name)
                self.type_sizes[name] = ent_n
                self.type_groups[name] = set()
                self.group_pos[name] = 0
                self.group_assignments[name] = np.ones(ent_n, dtype=np.int32)
                self.group_assignments[name][:] = NOT_ASSIGNED
        self.suffstats = {}
        self.component_dp_count = {}
        self.model = modeltype

    def get_axes_pos(self, type_name):
        ap = []
        for ei, (n, ent_n) in enumerate(self.relation_def):
            if n == type_name:
                ap.append(ei)
        return ap

    def create_group(self, type_name):
        """
        Return a group id, an opaque group handle
        """
        # create the group
        grp_id = self.group_pos[type_name]
        self.group_pos[type_name] += 1
        self.type_groups[type_name].add(grp_id)

        # what are the new groups to create? 
        grp_list = [self.type_groups[n] for n, s in self.relation_def]

        # create the suffstat blocks along every other axis
        new_ss = set(util.cart_prod(grp_list)) - set(self.suffstats.keys()) 
        # create all of them
        for k in new_ss:
            self.suffstats[k] = self.model.create_ss()
            self.component_dp_count[k] = 0
        return grp_id

    def delete_group(self, type_name, group_id):
        #
        self.type_groups[type_name].remove(group_id)
        
        grp_list = [self.type_groups[n] for n, s in self.relation_def]

        del_ss = set(self.suffstats.keys()) - set(util.cart_prod(grp_list))

        # delete those ss
        for d in del_ss:
            del self.suffstats[d]
            del self.component_dp_count[d]

    def get_all_groups(self, type_name):
        """
        Get all the groups for a given type
        """
        return self.type_groups[type_name]        
    
    def _get_groups(self, pos):
        """
        return the groups coordinates
        """
        ret = []
        for p, (name, ent_n) in zip(pos, self.relation_def):
            ret.append(self.group_assignments[name][p])
        return tuple(ret)
    
    def add_entity_to_group(self, type_name, group_id, entity_pos):
        """
        """

        uap = util.unique_axes_pos(self.get_axes_pos(type_name), entity_pos, 
                                   [range(g) for n, g in self.relation_def])
        assert self.group_assignments[type_name][entity_pos] == NOT_ASSIGNED

        self.group_assignments[type_name][entity_pos] = group_id

        for p in uap:
            group_coord = self._get_groups(p)
            v = self.data[p]
            if -1 not in group_coord: # FIXME is this the correct way? 
                cur_ss = self.suffstats[group_coord]
                self.suffstats[group_coord] = self.model.ss_add(cur_ss, self.hps, v)
                self.component_dp_count[group_coord] += 1
        
    def remove_entity_from_group(self, type_name, group_id, entity_pos):
        """
        """
        uap = util.unique_axes_pos(self.get_axes_pos(type_name), entity_pos, 
                                   [range(g) for n, g in self.relation_def])

        for p in uap:
            group_coord = self._get_groups(p)
            v = self.data[p]
            if -1 not in group_coord: # FIXME is this the correct way? 
                cur_ss = self.suffstats[group_coord]
                self.suffstats[group_coord] = self.model.ss_rem(cur_ss, self.hps,v)
                self.component_dp_count[group_coord] -= 1

        self.group_assignments[type_name][entity_pos] = -1
        
    def post_pred(self, type_name, group_id, entity_pos):
        """
        Prob of assigning this entity to this group for this type
        for the time being get total model score, 
        """
        

        # what's the right way to do this? FIXME
        oldscore = self.total_score()
        self.add_entity_to_group(type_name, group_id, entity_pos)
        newscore = self.total_score()
        self.remove_entity_from_group(type_name, group_id, entity_pos)
        return newscore - oldscore

    def total_score(self):
        """
        
        """
        score = 0
        for k in self.component_dp_count.keys():
            if self.component_dp_count[k] > 0:
                ss = self.suffstats[k]
                score += self.model.ss_score(ss, self.hps)

        return score

    def set_hps(self, hps):
        self.hps = hps


class TypeInterface(object):
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

    def create_group(self):
        """
        Returns group ID
        """
        rel_groupid = [r.create_group(t) for t, r in self.relations]
        
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
