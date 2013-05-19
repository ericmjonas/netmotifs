import numpy as np
import util

# GIBBS SAMPLING


def gibbs_sample_type(type_intf):

    T_N = type_intf.entity_count()

    for entity_pos in range(T_N):
        g = type_intf.remove_entity_from_group(entity_pos)
        if type_intf.group_size(g) == 0:
            temp_group = g
        else:
            temp_group = type_intf.create_group()

        groups = type_intf.get_groups()
        scores = np.zeros(len(groups))
        for gi, group_id in enumerate(groups):
            scores[gi] = type_intf.post_pred(group_id, entity_pos)
        #print entity_pos, scores
        sample_i = util.sample_from_scores(scores)
        new_group = groups[sample_i]

        type_intf.add_entity_to_group(new_group, entity_pos)
        if new_group != temp_group:
            type_intf.delete_group(temp_group)

def gibbs_sample_type_nonconj(type_intf, M):
    """
    Radford neal Algo 8 for non-conj models
    
    M is the number of ephemeral clusters
    
    We assume that every cluster in the model is currently used
    
    """
    T_N = type_intf.entity_count()

    for entity_pos in range(T_N):
        g = type_intf.remove_entity_from_group(entity_pos)
        extra_groups = []
        if type_intf.group_size(g) == 0:
            extra_groups.append(g)
        while len(extra_groups) < M:
            extra_groups.append(type_intf.create_group())

        groups = type_intf.get_groups()
        scores = np.zeros(len(groups))
        for gi, group_id in enumerate(groups):
            scores[gi] = type_intf.post_pred(group_id, entity_pos)
            # correct the score for the empty groups
            if group_id in extra_groups:
                scores[gi] -= np.log(M)
        #print entity_pos, scores
        sample_i = util.sample_from_scores(scores)
        new_group = groups[sample_i]

        type_intf.add_entity_to_group(new_group, entity_pos)
        for eg in extra_groups:
            if type_intf.group_size(eg) == 0:
                type_intf.delete_group(eg)

