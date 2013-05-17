import util


NOT_ASSIGNED = -1

class FastRelation(object):
    """
    Low-level implementaiton of a relation, optimized for performance
    
    WE create a dense matrix/cube/etc of suffstats under the hood, and then
    grow and or shrink. Sometimes we will need to grow / double in size
    (ala std::vectors) but that's ok. 
    
    group coordinates are tuples of ints

    Right now we assume data is fully dense
    
    """
    
    def __init__(self, domain_def, data, modeltype = None):

        assert len(domain_def) > 1
        
        self.data = data.flatten()

        self.model = modeltype

        self.axes = [dom_name for dom_name, size in domain_def]
        self.domains = {}
        for domain_name, domain_size in domain_def:
            self.domains[domain_name] = domain_size

        self.domain_to_axispos = {}
        for d in self.domains:
            self.domain_to_axispos[d] = tuple([i for n,i in zip(self.axes, range(len(self.axes))) if n == d])

        self.components = {}
        self.components_dp_count = {}

        self.group_id = 0

        self.domain_entity_assignment = {}
        self.domain_groups = {}
        for d in self.domains:
            self.domain_entity_assignment[d] = np.ones(self.domains[d], dtype=np.int32)*NOT_ASSIGNED
            self.domain_groups[d] = set()
        # initial setup 
        self.datapoint_groups = np.ones((len(self.data), 
                                         len(self.axes)), dtype=np.int32)
        self.datapoint_groups[:] = NOT_ASSIGNED
        self.datapoint_entity_index = np.zeros((len(self.data), 
                                         len(self.axes)), dtype=np.int32)

        self.entity_to_dp = {domain_name: {i : set() for i in range(N)} for domain_name, N in self.domains.iteritems()}
        # create rediciulous look-up table
        pos = 0
        for dp_coord in util.cart_prod([range(s) for n, s in domain_def]):
            for ax_pos, ax_name in zip(range(len(self.axes)), self.axes):
                entity_id = dp_coord[ax_pos]
                self.entity_to_dp[ax_name][entity_id].add(pos)
            self.datapoint_entity_index[pos] = dp_coord
            pos += 1

    def assert_unassigned(self):
        """
        Sanity check to make sure everything is unassigned
        """
        assert len(self.components) == 0
        assert len(self.components_dp_count) == 0
        for d in self.domains:
            assert (self.domain_entity_assignment[d] == NOT_ASSIGNED).all()
        assert (self.datapoint_groups[:] == NOT_ASSIGNED).all()
        
    def _get_axispos_for_domain(self, domain_name):
        return self.domain_to_axispos[domain_name]

    def _datapoints_for_entity(self, domain_name, entity_pos):
        return self.entity_to_dp[domain_name][entity_pos]

    def _get_dp_entity_coords(self, dp):
        return self.datapoint_entity_index[dp]
    def _get_dp_group_coords(self, dp):

        return self.datapoint_groups[dp]

    def _set_dp_group_coords(self, dp, group_coords):
        self.datapoint_groups[dp] = group_coords

    def _set_entity_group(self, domain_name, entity_pos, group_id):
        self.domain_entity_assignment[domain_name][entity_pos] = group_id

    def _get_entity_group(self, domain_name, entity_pos):
        return self.domain_entity_assignment[domain_name][entity_pos]

    def _get_data_value(self, dp):
        return self.data[dp]

    def create_group(self, domain_name):
        """
        Return a group ID, a unique group handle
        """
        new_gid = self.group_id
        self.domain_groups[domain_name].add(new_gid)

        # add to list of groups for this domain
        group_coords = util.unique_axes_pos(self._get_axispos_for_domain(domain_name), 
                                            new_gid, 
                                    [self.domain_groups[dn] for dn in self.axes])
                       
        for g in group_coords:
            self.components[g] = self.model.create_ss()
            self.components_dp_count[g] = 0
        self.group_id += 1
        return new_gid

    def delete_group(self, domain_name, group_id):
        """
        Delete a passed-in group id
        """
        # FIXME add snaity check here? 
        # add to list of groups for this domain
        group_coords = util.unique_axes_pos(self._get_axispos_for_domain(domain_name), 
                                        group_id, 
                                            [self.domain_groups[dn] for dn in self.axes])
                       
        for g in group_coords:
            del self.components[g] 
            del self.components_dp_count[g]
        self.domain_groups[domain_name].remove(group_id)

    def get_all_groups(self, domain_name):
        """
        Return a list of all the groups for this domain
        """
        return self.domain_groups[domain_name]

    def add_entity_to_group(self, domain_name, group_id, entity_pos):
        """
        It might be the case that a given datapoint is ALRADY at the 
        group
        """
        for dp in self._datapoints_for_entity(domain_name, entity_pos):
            value = self._get_data_value(dp)
            current_group_coords = self._get_dp_group_coords(dp)

            new_group_coords = list(current_group_coords)
            dp_entity_pos = self._get_dp_entity_coords(dp)
            
            for axis_pos in self._get_axispos_for_domain(domain_name):
                if dp_entity_pos[axis_pos] == entity_pos:
                    new_group_coords[axis_pos] = group_id

            if NOT_ASSIGNED in current_group_coords and NOT_ASSIGNED not in new_group_coords:
                self.components[tuple(new_group_coords)] = self.model.ss_add(self.components[tuple(new_group_coords)], self.hps, value)
                self.components_dp_count[tuple(new_group_coords)] += 1

            self._set_dp_group_coords(dp, new_group_coords)
        self._set_entity_group(domain_name, entity_pos, group_id)

    def remove_entity_from_group(self, domain_name, group_id, entity_pos):
        """
        """

        for dp in self._datapoints_for_entity(domain_name, entity_pos):
            value = self._get_data_value(dp)
            group_coords = self._get_dp_group_coords(dp)
            if NOT_ASSIGNED not in group_coords:

                self.components[tuple(group_coords)] = self.model.ss_rem(self.components[tuple(group_coords)], self.hps, value)
                self.components_dp_count[tuple(group_coords)] -= 1

            for axis_pos in self._get_axispos_for_domain(domain_name):
                group_coords[axis_pos] = NOT_ASSIGNED
            self._set_dp_group_coords(dp, group_coords)
        self._set_entity_group(domain_name, entity_pos, NOT_ASSIGNED)


    def post_pred(self, domain_name, group_id, entity_pos):
        score = 0.0
        for dp in self._datapoints_for_entity(domain_name, entity_pos):
            data_value = self._get_data_value(dp)
            current_group_coords = self._get_dp_group_coords(dp)
            new_group_coords = list(current_group_coords)
            dp_entity_pos = self._get_dp_entity_coords(dp)
            
            for axis_pos in self._get_axispos_for_domain(domain_name):
                if dp_entity_pos[axis_pos] == entity_pos:
                    new_group_coords[axis_pos] = group_id
            ss = self.components[tuple(new_group_coords)]
                                 
            inc_score = self.model.post_pred(ss, 
                                             self.hps, data_value)
            score += inc_score
        return score
        
    def total_score(self):
        score = 0
        pos = 0
        for k in self.components_dp_count.keys():
            pos += 1
            if self.components_dp_count[k] > 0:
                ss = self.components[k]
                score += self.model.ss_score(ss, self.hps)

        return score


    def set_hps(self, hps):
        self.hps = hps

