import numpy as np
from copy import deepcopy

def addelement(partlist, e):
    newpartlist = []
    for part in partlist:
        npart = part + [[e]]
        newpartlist += [npart]
        for i in xrange(len(part)):
            npart = deepcopy(part)
            npart[i] += [e]
            newpartlist += [npart]
    return newpartlist

def partition(n):
    if n == 0: return []
    partlist = [[[1]]]
    for i in xrange(2, n+1):
        partlist = addelement(partlist, i)
    return partlist

def canonicalize_assignment(assignments):
    """
    Canonicalize an assignment vector. this works as follows:
    largest group is group 0, 
    next largest is group 1, etc. 

    For two identically-sized groups, The lower one is
    the one with the smallest row
    
    """
    groups = {}
    for gi, g in enumerate(assignments):
        if g not in groups:
            groups[g] = []
        groups[g].append(gi)
    orig_ids = np.array(groups.keys())
    sizes = np.array([len(groups[k]) for k in orig_ids])

    unique_sizes = np.sort(np.unique(sizes))[::-1]
    out_assign = np.zeros(len(assignments), dtype=np.uint32)

    # unique_sizes is in big-to-small
    outpos = 0
    for size in unique_sizes:
        # get the groups of this size
        tgt_ids = orig_ids[sizes == size]
        minvals = [np.min(groups[tid]) for tid in tgt_ids]
        min_idx = np.argsort(minvals)
        for grp_id  in tgt_ids[min_idx]:
            out_assign[groups[grp_id]] = outpos
            outpos +=1 
    return out_assign

def part_to_assignvect(part, N):
    """
    partition [[1, 2, 3], [4, 5], [6]]
    to assignvect [0 0 0 1 1 2]
    """
    outvect = np.zeros(N, dtype=np.uint32)
    i = 0
    for pi, p in enumerate(part):
        outvect[np.array(p)-1] = pi
    return outvect


def enumerate_canonical_partitions(Nrows):
    parts = partition(Nrows)
    assignments = np.zeros((len(parts), Nrows), 
                            dtype = np.uint8)
    for pi, part in enumerate(parts):
        a = part_to_assignvect(part, Nrows)
        ca = canonicalize_assignment(a)
        assignments[pi] = ca
    return parts, assignments

def enumerate_possible_latents(domains):
    """
    domains is a list of dom sizes

    The enumeration is a listing of assignment vectors for each
    
    """
    if len(domains) == 1:
        parts, assignments = enumerate_canonical_partitions(domains[0])
        for a in assignments:
            yield (tuple(a), )
    elif len(domains) == 2:
        p1, a1 = enumerate_canonical_partitions(domains[0])
        p2, a2 = enumerate_canonical_partitions(domains[1])
        for a_1 in a1:
            for a_2 in a2:
                yield (tuple(a_1), tuple(a_2))

    else:
        raise NotImplementedError()
    
# def type_to_assign_vect(type_intf, av, rng=None):
#     """
#     for a given assignment vector, force the type into that format
#     """
#     assert len(av) == type_intf.entity_count()
#     id_to_gid = {}

#     for ei, a in enumerate(av):
#         oldgroup = type_intf.remove_entity_from_group(ei)
#         if type_intf.group_size(oldgroup) == 0:
#             type_intf.delete_group(oldgroup)
#         if a not in id_to_gid:
#             new_gid = type_intf.create_group(rng)
#             id_to_gid[a] = new_gid
#         type_intf.add_entity_to_group(id_to_gid[a], ei)
