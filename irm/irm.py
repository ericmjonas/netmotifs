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
                self.group_assignments[name] = np.ones(ent_n, dtype=np.int32)*-1
        self.suffstats = {}
        
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
        return grp_id

    def delete_group(self, type_name, group_id):
        #
        self.type_groups[type_name].remove(group_id)
        
        grp_list = [self.type_groups[n] for n, s in self.relation_def]

        del_ss = set(self.suffstats.keys()) - set(util.cart_prod(grp_list))

        # delete those ss
        for d in del_ss:
            del self.suffstats[d]

    def get_all_groups(self, type_name):
        """
        Get all the groups for a given type
        """
        return self.type_groups[type_name]        
    
    def get_groups(self, pos):
        """
        return the groups
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
        assert self.group_assignments[type_name][entity_pos] == -1

        self.group_assignments[type_name][entity_pos] = group_id

        for p in uap:
            group_coord = self.get_groups(p)
            v = self.data[p]
            if -1 not in group_coord: # FIXME is this the correct way? 
                cur_ss = self.suffstats[group_coord]
                self.suffstats[group_coord] = self.model.ss_add(cur_ss, self.hps, v)
        
    def remove_entity_from_group(self, type_name, group_id, entity_pos):
        """
        """
        uap = util.unique_axes_pos(self.get_axes_pos(type_name), entity_pos, 
                                   [range(g) for n, g in self.relation_def])

        for p in uap:
            group_coord = self.get_groups(p)
            v = self.data[p]
            cur_ss = self.suffstats[group_coord]
            self.suffstats[group_coord] = self.model.rem_ss(cur_ss, self.hps,v)

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
        scores = [self.model.ss_score(ss, self.hps) for ss in self.suffstats.values()]
        return np.sum(scores)

    def set_hps(self, hps):
        self.hps = hps
