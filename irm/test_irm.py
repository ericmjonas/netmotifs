from nose.tools import * 
import numpy as np
import models
import irm


"""
Cases to worry about

T1 x T2 
T1 x T2 x T3
T1 x T1 
T1 x T2 x T1
T1 x T1 x T2
T1 x T1 x T2 x T2

"""

def test_relation_T1T2_allone_singleton():
    T1_N = 3
    T2_N = 4

    data = np.arange(T1_N*T2_N)
    data.shape = T1_N, T2_N

    model =  models.AccumModel()
    r = irm.Relation([('T1', T1_N), ('T2', T2_N)], 
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
    assert_equal(s, 0.3)
    
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
    assert_equal(s, np.sum(data) + 0.3*T1_N*T2_N)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
        
        
def test_relation_T1T1_allone():
    T1_N = 10

    data = np.arange(T1_N * T1_N)
    data.shape = T1_N, T1_N

    model =  models.AccumModel()
    r = irm.Relation([('T1', T1_N), ('T1', T1_N)], 
                     data,model)
    hps = model.create_hps()
    hps['offset'] = 0.3

    r.set_hps(hps)
    
    t1_grp = r.create_group('T1')

    for i in range(T1_N):
        r.add_entity_to_group('T1', t1_grp, i)
    
    s = r.total_score()
    assert_equal(s, np.sum(data) + 0.3)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
    # remove everything
    for i in range(T1_N):
        r.remove_entity_from_group('T1', t1_grp, i)

    s = r.total_score()
    assert_equal(s, 0.3)
    
    #### DELETE THE GROUPS
    r.delete_group('T1', t1_grp)

    #### ADD TO SINGLETONS
    t1_grps = [r.create_group('T1') for _ in range(T1_N)]

    for i in range(T1_N):
        r.add_entity_to_group('T1', t1_grps[i], i)
    
    s = r.total_score()
    np.testing.assert_approx_equal(s, np.sum(data) + 0.3*T1_N*T1_N)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
