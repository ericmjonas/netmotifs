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

