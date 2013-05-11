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

def test_relation():
    T1_N = 3
    T2_N = 4

    data = np.ones((T1_N, T2_N))

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
    assert_equal(s, T1_N*T2_N + 0.3)
    # this should be the score for a mixture model with
    # simply these hypers and these sets of whatever
    
        
        
