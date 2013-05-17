from nose.tools import * 
import numpy as np
from numpy.testing import assert_approx_equal, assert_array_equal

import models
import irm
import relation
import util

"""
Cases to worry about

T1 x T2 
T1 x T2 x T3
T1 x T1 
T1 x T2 x T1
T1 x T1 x T2
T1 x T1 x T2 x T2

"""
def test_relation_T1T2_allone_singleton_default():
    relation_T1T2_allone_singleton(relation.Relation)

def test_relation_T1T2_allone_singleton_default():
    relation_T1T2_allone_singleton(relation.FastRelation)

def relation_T1T2_allone_singleton(relation_class):
    T1_N = 3
    T2_N = 4

    data = np.arange(T1_N*T2_N)
    data.shape = T1_N, T2_N

    model =  models.AccumModel()
    r = relation_class([('T1', T1_N), ('T2', T2_N)], 
                     data,model)
    hps = model.create_hps()
    hps['offset'] = 0.3

    r.set_hps(hps)
    
    t1_grp = r.create_group('T1')
    t2_grp = r.create_group('T2')
    

    for i in range(T1_N):
        r.add_entity_to_group('T1', t1_grp, i)


    for i in range(T2_N):
        r.add_entity_to_group('T2', t2_grp, i)
    
    s = r.total_score()
    assert_equal(s, np.sum(data) + 0.3)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
    # remove everything
    for i in range(T1_N):
        r.remove_entity_from_group('T1', t1_grp, i)
    for i in range(T2_N):
        r.remove_entity_from_group('T2', t2_grp, i)

    s = r.total_score()
    assert_equal(s, 0.0)
    
    #### DELETE THE GROUPS
    r.delete_group('T1', t1_grp)
    r.delete_group('T2', t2_grp)

    #### ADD TO SINGLETONS
    t1_grps = [r.create_group('T1') for _ in range(T1_N)]
    t2_grps = [r.create_group('T2') for _ in range(T2_N)]

    for i in range(T1_N):
        r.add_entity_to_group('T1', t1_grps[i], i)
    for i in range(T2_N):
        r.add_entity_to_group('T2', t2_grps[i], i)
    
    s = r.total_score()
    assert_approx_equal(s, np.sum(data) + 0.3*T1_N*T2_N)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
        
def test_relation_T1T1_allone_slow():
    relation_T1T1_allone(relation.Relation)

def test_relation_T1T1_allone_fast():
    relation_T1T1_allone(relation.FastRelation)
        
def relation_T1T1_allone(relation_class):
    T1_N = 10

    data = np.arange(T1_N * T1_N)
    data.shape = T1_N, T1_N

    model =  models.AccumModel()
    r = relation_class([('T1', T1_N), ('T1', T1_N)], 
                     data,model)
    hps = model.create_hps()
    hps['offset'] = 0.3

    r.set_hps(hps)
    
    t1_grp = r.create_group('T1')

    for i in range(T1_N):
        print "Test for T1 adding", t1_grp, i
        r.add_entity_to_group('T1', t1_grp, i)
    
    s = r.total_score()
    assert_approx_equal(s, np.sum(data) + 0.3)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
    # remove everything
    for i in range(T1_N):
        r.remove_entity_from_group('T1', t1_grp, i)

    s = r.total_score()
    assert_equal(s, 0.0)
    
    #### DELETE THE GROUPS
    r.delete_group('T1', t1_grp)

    print "SINGLETON TEST", "="*50
    #### ADD TO SINGLETONS
    t1_grps = [r.create_group('T1') for _ in range(T1_N)]
    
    for i in range(T1_N):
        print "TEST adding entity", i, "to group",  t1_grps[i], 
        r.add_entity_to_group('T1', t1_grps[i], i)
    
    s = r.total_score()
    np.testing.assert_approx_equal(s, np.sum(data) + 0.3*T1_N*T1_N)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever

def test_type_if_rel():
    type_if_rel(relation.Relation)

def test_type_if_rel_fast():
    type_if_rel(relation.FastRelation)

def type_if_rel(relation_class):
    """
    Test if transactions on the type interface propagate 
    correctly to relations and preserve invariants
    """
    
    T1_N = 2
    T2_N = 2

    data = np.arange(T1_N * T2_N)
    data.shape = T1_N, T2_N


    model =  models.AccumModel()
    r = relation_class([('T1', T1_N), ('T2', T2_N)], 
                     data,model)
    hps = model.create_hps()
    hps['offset'] = 0.3

    r.set_hps(hps)

    tf_1 = irm.TypeInterface(T1_N, [('T1', r)])
    tf_1.set_hps(1.0)
    tf_2 = irm.TypeInterface(T2_N, [('T2', r)])
    tf_2.set_hps(1.0)

    assert_array_equal(tf_1.get_assignments(), np.ones(T1_N)*irm.NOT_ASSIGNED)


    ### put all T2 into one group, T1 in singletons
    t2_g1 = tf_2.create_group()
    assert_equal(len(r.get_all_groups('T2')), 1)
    for i in range(T2_N):
        tf_2.add_entity_to_group(t2_g1, i)
    
    t1_grps= [tf_1.create_group() for _ in range(T1_N)]
    [tf_1.add_entity_to_group(g, i) for g, i in zip(t1_grps, range(T1_N))]
    
    # total score 
    assert_approx_equal(r.total_score(), 
                        np.sum(np.sum(data, axis=1)) + 0.3*T1_N)

    #Now remove the last entity from the last group in T1 and compute
    #post pred
    tf_1.remove_entity_from_group(T1_N -1)
    for g_i, g in enumerate(t1_grps[:-1]):
        print "-"*70
        score = tf_1.post_pred(g, T1_N-1)
        est_score_old = np.sum(data[g_i, :]) 

        dp = np.hstack([data[g_i, :], data[T1_N -1, :]])
        print dp
        est_score_new = np.sum(dp) 
        score_delta = est_score_new- est_score_old + util.crp_post_pred(1, T1_N, 1.0)
        assert_approx_equal(score_delta, score)

