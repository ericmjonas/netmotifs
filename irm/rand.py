import numpy as np
import scipy
import scipy.misc
from nose.tools import * 
#import pyximport; pyximport.install()
#import fastrand

def canonicalize_assignment_vector(x):
    """
    Take in an assignment vector and redo the assignments such that
    the assignment values are monotonic from 0 to GRPMAX
    """
    u = np.unique(x)
    lut = {}
    for ui, u in enumerate(u):
        lut[u] = ui

    res_vector = np.zeros_like(x)
    for xi, xv in enumerate(x):
        res_vector[xi] = lut[xv]

    return res_vector


def assignment_vector_to_list_of_sets(x):
    """
    """
    u = np.unique(x)
    lut = {}
    
    for ui, u in enumerate(x):
        if u not in lut:
            lut[u] = set()
        lut[u].add(ui)

    # turn into list
    return lut.values()



def compute_adj_rand_index(ground_truth_partition, found_partition):
    '''
Computes the adjusted rand index of the groups in the found partition as
compared to the ground truth partition. Both partitions should be a
canonical mapping such that
partition[i] = group containing item i (None if in no group)
'''

    assert len(ground_truth_partition) == len(found_partition)

    # replace any Nones with the next available group id
    no_assignment_id = max(ground_truth_partition + found_partition) + 1
    for part in [ground_truth_partition, found_partition]:
        for i in range(len(part)):
            if part[i] == None:
                part[i] = no_assignment_id

    assert all([x != None for x in found_partition])
    assert all([x != None for x in ground_truth_partition])

    num_ground_truth_groups = len(set(ground_truth_partition))
    num_found_groups = len(set(found_partition))

    # These two edge cases cause a divide-by-zero error if the ground truth
    # and found partitions are identical. Don't bother to calculate.
    if (((num_found_groups == 1) and (num_ground_truth_groups == 1))
        or ((num_found_groups == len(ground_truth_partition))
            and num_ground_truth_groups == len(ground_truth_partition))):

        return 1.0


    contingency_table = np.zeros((num_found_groups,
                                     num_ground_truth_groups))
    

    for item, gt_group in enumerate(ground_truth_partition):
        found_group = found_partition[item]
        contingency_table[found_group, gt_group] += 1

    # For more details on this algorithm (since this code is not the most
    # readable or best named ever), see
    # http://faculty.washington.edu/kayee/pca/supp.pdf
    # or http://en.wikipedia.org/wiki/Adjusted_rand_index
    all_entries = np.sum(scipy.misc.comb(contingency_table, 2))
    rows_collapsed = np.sum(scipy.misc.comb(np.sum(contingency_table, 0), 2))
    cols_collapsed = np.sum(scipy.misc.comb(np.sum(contingency_table, 1), 2))
    num_items = scipy.misc.comb(len(ground_truth_partition), 2)

    ari = ( (all_entries - (rows_collapsed * cols_collapsed / num_items))
           / ( ((rows_collapsed + cols_collapsed) / 2)
              - ((rows_collapsed * cols_collapsed) / num_items)))
    assert not np.isnan(ari)

    return ari


def test_ari():
    assert_almost_equal(compute_adj_rand_index([0, 0, 0, 1, 1, 1,2,2,2],
                                              [1, 1, 1, 2, 2, 2, 0, 0, 0]),
                        1.0, 2)
    

def twocomb(x):
    """
    compute binom(x, 2)
    """

    return  x*(x-1) / 2. 

    
def compute_adj_rand_index_fast(list_of_sets_U, list_of_sets_V):
    a_i = np.array([len(x) for x in list_of_sets_U])
    b_i = np.array([len(x) for x in list_of_sets_V])
    
    
    ctable = np.zeros((len(list_of_sets_U), len(list_of_sets_V)),
                      dtype=np.uint32)

    for ui, u in enumerate(list_of_sets_U):
        for vi, v in enumerate(list_of_sets_V):
            ctable[ui, vi] = len(u & v)

    all_entries = np.sum(twocomb(np.array(ctable.flat)))

    aisum = np.sum(twocomb(a_i))
    bjsum = np.sum(twocomb(b_i))

    sc =twocomb(np.sum(a_i))

    num = float(all_entries) - (aisum * bjsum / sc)
    den = 0.5 * (aisum + bjsum) - (aisum * bjsum / sc)

    return num/den



def create_data(groups, rows_per_group):
    data = range(groups) * rows_per_group

    dataa = np.array(data, dtype=np.uint32)
    return np.random.permutation(dataa)
    

def test_rands():
    
    for groups in [10, 100, 500]:
        for rows_per_group in [10, 50, 100]:
            d1 = create_data(groups, rows_per_group)
        
            d2 = create_data(groups, rows_per_group)

            s1 = assignment_vector_to_list_of_sets(d1)
            s2 = assignment_vector_to_list_of_sets(d2)
            r1 = compute_adj_rand_index_fast(s1, s2)

            r2 = fastrand.compute_adj_rand(d1, d2)
            assert_almost_equal(r1, r2, 2)
            

def compute_similarity_stats(c1, c2):
    """
    Compute the similarity statistics for two clusterings
    """
    assert len(c1) == len(c2)
    
    N = len(c1)
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0
    
    
    for i1 in range(N):
        for i2 in range(N):
            if i1 == i2:
                continue
            a1_c1 = c1[i1]
            a2_c1 = c1[i2]
            a1_c2 = c2[i1]
            a2_c2 = c2[i2]
            
            if a1_c1 == a2_c1 and a1_c2 == a2_c2:
                n_11 +=1
            elif a1_c1 != a2_c1 and a1_c2 != a2_c2:
                n_00 += 1
            elif a1_c1 == a2_c1 and a1_c2 != a2_c2:
                n_10 += 1
            elif a1_c1 != a2_c1 and a1_c2 == a2_c2:
                n_01 += 1
                
    return {'n00' : n_00/2, 
            'n01' : n_01/2, 
            'n10' : n_10/2, 
            'n11' : n_11/2}

def compute_jaccard(c1, c2):
    ss = compute_similarity_stats(c1, c2)
    return float(ss['n11']) / (ss['n11'] + ss['n01'] + ss['n10'])

