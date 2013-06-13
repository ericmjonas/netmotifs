from nose.tools import * 
import numpy as np
from matplotlib import pylab


import irm
from irm import util 
from irm import models, model
from irm import Relation

def test_get_components():
    """
    
    """
    T1_N = 10
    T2_N = 20
    np.random.seed(0)
    rng = irm.RNG()

    data = np.random.rand(T1_N, T2_N) > 0.5
    data.shape = T1_N, T2_N

    m =  models.BetaBernoulli()
    r = Relation([('T1', T1_N), ('T2', T2_N)], 
                     data,m)
    hps = m.create_hps()
    hps['alpha'] = 1.0
    hps['beta'] = 1.0

    r.set_hps(hps)

    tf_1 = model.DomainInterface(T1_N, [('T1', r)])
    tf_1.set_hps(1.0)
    tf_2 = model.DomainInterface(T2_N, [('T2', r)])
    tf_2.set_hps(1.0)

    T1_GRPN = 4
    t1_assign = np.arange(T1_N) % T1_GRPN
    t1_grps = {}
    for i, gi in enumerate(t1_assign):
        if gi not in t1_grps:
            g = tf_1.create_group(rng)
            t1_grps[gi] = g
        tf_1.add_entity_to_group(t1_grps[gi], i)

    T2_GRPN = 4
    t2_assign = np.arange(T2_N) % T2_GRPN
    t2_grps = {}
    for i, gi in enumerate(t2_assign):
        if gi not in t2_grps:
            g = tf_2.create_group(rng)
            t2_grps[gi] = g
        tf_2.add_entity_to_group(t2_grps[gi], i)

    t1_assign_g = tf_1.get_assignments()
    t2_assign_g = tf_2.get_assignments()
    
    for t1_g in np.unique(t1_assign_g):
        for t2_g in np.unique(t2_assign_g):
            t1_entities = np.argwhere(t1_assign_g == t1_g).flatten()
            t2_entities = np.argwhere(t2_assign_g == t2_g).flatten()
            
            dps = []
            for e1 in t1_entities:
                for e2 in t2_entities:
                    dps.append(data[e1, e2])
            heads = np.sum(np.array(dps)==1)
            tails = np.sum(np.array(dps)==0)
            c = r.get_component((tf_1.get_relation_groupid(0, t1_g), 
                                 tf_2.get_relation_groupid(0, t2_g)))
            assert_equal(heads, c['heads'])
            assert_equal(tails, c['tails'])
                                    


def test_set_components():
    """
    
    """
    T1_N = 10
    T2_N = 20
    np.random.seed(0)
    rng = irm.RNG()

    data = np.random.rand(T1_N, T2_N) > 0.5
    data.shape = T1_N, T2_N

    m =  models.BetaBernoulli()
    r = Relation([('T1', T1_N), ('T2', T2_N)], 
                     data,m)
    hps = m.create_hps()
    hps['alpha'] = 1.0
    hps['beta'] = 1.0

    r.set_hps(hps)

    tf_1 = model.DomainInterface(T1_N, [('T1', r)])
    tf_1.set_hps(1.0)
    tf_2 = model.DomainInterface(T2_N, [('T2', r)])
    tf_2.set_hps(1.0)

    T1_GRPN = 4
    t1_assign = np.arange(T1_N) % T1_GRPN
    t1_grps = {}
    for i, gi in enumerate(t1_assign):
        if gi not in t1_grps:
            g = tf_1.create_group(rng)
            t1_grps[gi] = g
        tf_1.add_entity_to_group(t1_grps[gi], i)

    T2_GRPN = 4
    t2_assign = np.arange(T2_N) % T2_GRPN
    t2_grps = {}
    for i, gi in enumerate(t2_assign):
        if gi not in t2_grps:
            g = tf_2.create_group(rng)
            t2_grps[gi] = g
        tf_2.add_entity_to_group(t2_grps[gi], i)

    t1_assign_g = tf_1.get_assignments()
    t2_assign_g = tf_2.get_assignments()

    allmodel = model.IRM({'T1' : tf_1, 'T2' : tf_2}, 
                         {'R1' : r})

    lastscore = allmodel.total_score()
    for t1_g in np.unique(t1_assign_g):
        for t2_g in np.unique(t2_assign_g):
            t1_entities = np.argwhere(t1_assign_g == t1_g).flatten()
            t2_entities = np.argwhere(t2_assign_g == t2_g).flatten()
            
            dps = []
            for e1 in t1_entities:
                for e2 in t2_entities:
                    dps.append(data[e1, e2])
            heads = np.sum(np.array(dps)==1)
            tails = np.sum(np.array(dps)==0)
            # check if the current value is correct
            c = r.get_component((tf_1.get_relation_groupid(0, t1_g), 
                                 tf_2.get_relation_groupid(0, t2_g)))
            assert_equal(heads, c['heads'])
            assert_equal(tails, c['tails'])
                                    
            # now we set them to a random value
            c = r.set_component((tf_1.get_relation_groupid(0, t1_g), 
                                 tf_2.get_relation_groupid(0, t2_g)),
                                {'heads' : int(heads), 
                                 'tails' : int(tails) + 1})
            
            assert allmodel.total_score() != lastscore
            lastscore = allmodel.total_score()

            c = r.set_component((tf_1.get_relation_groupid(0, t1_g), 
                                 tf_2.get_relation_groupid(0, t2_g)),
                                {'heads' : int(heads) + 1, 
                                 'tails' : int(tails) + 1})
            
            assert allmodel.total_score() != lastscore
            lastscore = allmodel.total_score()
