from nose.tools import * 
import numpy as np
from numpy.testing import assert_approx_equal, assert_array_equal, assert_array_almost_equal

import irm
from irm import models
from irm import relation
from irm import util
from irm import model
from irm import Relation
from irm import pyirm
import time

RELATIONS = [
             ('cpp', irm.pyirmutil.Relation), 
             ('parcpp', irm.pyirmutil.ParRelation)]


def gen_fake_data(m, N):
    """
    Generate some simple fake data, we really should centralize this a bit more
    """

    if m == models.BetaBernoulli():
        data = (np.random.rand(N) > 0.5).astype(np.uint8)        
    elif m == models.LogisticDistance():
        data = np.zeros(N, dtype=m.data_dtype())
        
    else:
        raise NotImplemented("model Type unknown")
    return data

def assert_all_approx_same(x):
    """
    assert all entries in the array are roughly the same
    """
    x_a = np.array(x)
    a = np.ones_like(x_a)
    a[:] = x_a[0]
    assert_array_almost_equal(a, x_a, 2)

def relation_r_set(rel, latent, rng):
    """
    put relation in configuration, return the mapping of ids to groupids
    """

    domains = latent['domains'].keys()
    d_groups = {}
    for dn in domains:
        d_groups[dn] = {}
        for ai, a in enumerate(latent['domains'][dn]['assignment']):
            if a not in d_groups[dn]:
                gid = rel.create_group(dn, rng)
                d_groups[dn][a] = gid
            rel.add_entity_to_group(dn, d_groups[dn][a], ai)
    return d_groups

def relation_r_set_comps(rel, domain_tuple, latent_ss, d_groups):
    for ss_gc, ss_val in latent_ss.iteritems():
        new_coord = []
        for dn, pos in zip(domain_tuple, ss_gc):
            new_coord.append(d_groups[dn][pos])

        n_tuple = tuple(new_coord)

        rel.set_component(n_tuple, ss_val)
        
"""
The python model implementations don't properly handle what we consider
to be "Contemporary" nonconjugate model interfaces. 

BetaBernoulliNonConj in python wants to keep track of heads/tails 
in the suffstats vector instead of JUST the parameter, and then the
SCORE method of the python models doesn't pass in a list of data. 

This is a pretty substantial deviation and is very frustrating. 

We'd have to change a lot of things to make it, I don't know, work? 

So while I really want this code to exist, there's a big difference
in effective APIs and it doens't quite seem worth while under
deadline to bring it up to speed. 

I'm flagging this because IT WILL COME BACK TO BITE US IN THE ASS. 


"""


def test_total_score_T1_T2():

    rng = pyirm.RNG()
    np.random.seed(0)
    for modelstr in ['BetaBernoulliNonConj']:
        
        for T1_N in [1, 5, 10]:
            for T2_N in [1, 7, 10]:
                data = {'domains' : {'d1': {'N' : T1_N}, 
                                     'd2' : {'N' : T2_N}}, 
                        'relations' : {'r1' : {'relation' : ('d1', 'd2'), 
                                               'model' : modelstr}}}

                latent = {'domains' : 
                          {'d1' : 
                           {'assignment' : np.random.permutation(T1_N) % 3}},
                          'd2' : 
                          {'assignment' : np.random.permutation(T2_N) % 3}}

                n_latent, n_data = irm.data.synth.prior_generate(latent, data)


                m =  models.NAMES[modelstr]()

                scores = {}

                for relation_name, relation_class in RELATIONS:

                    rel_data = n_data['relations']['r1']['data']

                    if rel_data.dtype == np.bool:
                        rel_data = rel_data.astype(np.uint8)


                    rel_hps = n_latent['relations']['r1']['hps']

                    r = relation_class([('d1', T1_N), ('d2', T2_N)], 
                                       rel_data, m)
                    r.set_hps(rel_hps)

                    d_groups = relation_r_set(r, n_latent, rng)

                    if not m.conjugate():
                        relation_r_set_comps(r, 
                                             n_data['relations']['r1']['relation'], 
                                             n_latent['relations']['r1']['ss'], 
                                             d_groups)

                    scores[relation_name] = r.total_score()
                print scores
                assert_all_approx_same(scores.values())


def test_total_score_T1_T1():

    rng = pyirm.RNG()
    np.random.seed(0)
    for modelstr in ['BetaBernoulliNonConj', 'LogisticDistance']:
        
        for T1_N in [1, 5, 10, 50]:
            data = {'domains' : {'d1': {'N' : T1_N}}, 
                    'relations' : {'r1' : {'relation' : ('d1', 'd1'), 
                                           'model' : modelstr}}}

            latent = {'domains' : 
                      {'d1' : 
                       {'assignment' : np.random.permutation(T1_N) % 3}}}

            n_latent, n_data = irm.data.synth.prior_generate(latent, data)


            m =  models.NAMES[modelstr]()

            scores = {}

            for relation_name, relation_class in RELATIONS:

                rel_data = n_data['relations']['r1']['data']

                if rel_data.dtype == np.bool:
                    rel_data = rel_data.astype(np.uint8)


                rel_hps = n_latent['relations']['r1']['hps']

                r = relation_class([('d1', T1_N), ('d1', T1_N)], 
                                   rel_data, m)
                r.set_hps(rel_hps)

                d_groups = relation_r_set(r, n_latent, rng)

                if not m.conjugate():
                    relation_r_set_comps(r, 
                                         n_data['relations']['r1']['relation'], 
                                         n_latent['relations']['r1']['ss'], 
                                         d_groups)

                scores[relation_name] = r.total_score()
            print scores
            assert_all_approx_same(scores.values())


def compare_post_pred(pp1, pp2):
    for ei in pp1.keys():
        for tgt_grp in pp1[ei].keys():
            assert_approx_equal(pp1[ei][tgt_grp], 
                                pp2[ei][tgt_grp])

        
    
def test_postpred_T1_T1():

    rng = pyirm.RNG()
    np.random.seed(0)
    for modelstr in ['BetaBernoulliNonConj', 'LogisticDistance']:
        
        for T1_N in [4]:
            data = {'domains' : {'d1': {'N' : T1_N}}, 
                    'relations' : {'r1' : {'relation' : ('d1', 'd1'), 
                                           'model' : modelstr}}}

            latent = {'domains' : 
                      {'d1' : 
                       {'assignment' : np.random.permutation(T1_N) % 3}}}

            n_latent, n_data = irm.data.synth.prior_generate(latent, data)


            m =  models.NAMES[modelstr]()

            postpred = {}
            postpred_map = {}
            for relation_name, relation_class in RELATIONS:

                rel_data = n_data['relations']['r1']['data']

                if rel_data.dtype == np.bool:
                    rel_data = rel_data.astype(np.uint8)


                rel_hps = n_latent['relations']['r1']['hps']

                r = relation_class([('d1', T1_N), ('d1', T1_N)], 
                                   rel_data, m)
                r.set_hps(rel_hps)

                d_groups = relation_r_set(r, n_latent, rng)

                if not m.conjugate():
                    relation_r_set_comps(r, 
                                         n_data['relations']['r1']['relation'], 
                                         n_latent['relations']['r1']['ss'], 
                                         d_groups)

                # for each entity, we remove it, compute the post-pred across both each
                
                # invert
                new_old = {v : k for k, v in d_groups['d1'].iteritems()}

                # group individually 
                rpp = {}
                rpp_map = {}
                for ei in range(T1_N):
                    orig_grp = latent['domains']['d1']['assignment'][ei]
                    gi = d_groups['d1'][orig_grp]
                    r.remove_entity_from_group('d1', gi, ei)
                    # post pred on others
                    rpp[ei] = {}
                    for gi_tgt in np.unique(d_groups['d1']):
                        pp = r.post_pred('d1', gi_tgt, ei)
                        # get original latent groupid for comparison
                        orig_gi = new_old[gi_tgt]
                        rpp[ei][orig_gi] = pp 
                    # order
                    gs = new_old.keys()
                    rpp_map[ei]= dict(zip([new_old[g] for g in gs], 
                                          r.post_pred_map('d1', 
                                                          gs, 
                                          ei)))
                    # put it back
                    r.add_entity_to_group('d1', gi, ei)
                postpred[relation_name] = rpp
                postpred_map[relation_name] = rpp_map


                
            compare_post_pred(postpred['cpp'], 
                              postpred['parcpp'])

            compare_post_pred(postpred['parcpp'], 
                              postpred_map['parcpp'])

            compare_post_pred(postpred['cpp'], 
                              postpred_map['parcpp'])

def test_postpred_T1_T1_speed():
    """ 
    Hack speed test
    """

    rng = pyirm.RNG()
    np.random.seed(0)
    for modelstr in ['LogisticDistance', 'ExponentialDistancePoisson']:
        
        for T1_N in [1000]:
            GROUP_N = 30 
            data = {'domains' : {'d1': {'N' : T1_N}}, 
                    'relations' : {'r1' : {'relation' : ('d1', 'd1'), 
                                           'model' : modelstr}}}

            latent = {'domains' : 
                      {'d1' : 
                       {'assignment' : np.random.permutation(T1_N) % GROUP_N}}}

            n_latent, n_data = irm.data.synth.prior_generate(latent, data)


            m =  models.NAMES[modelstr]()

            postpred = {}
            postpred_map = {}
            map_times = {}
            for relation_name, relation_class in RELATIONS:

                rel_data = n_data['relations']['r1']['data']

                if rel_data.dtype == np.bool:
                    rel_data = rel_data.astype(np.uint8)


                rel_hps = n_latent['relations']['r1']['hps']

                r = relation_class([('d1', T1_N), ('d1', T1_N)], 
                                   rel_data, m)
                r.set_hps(rel_hps)

                d_groups = relation_r_set(r, n_latent, rng)

                if not m.conjugate():
                    relation_r_set_comps(r, 
                                         n_data['relations']['r1']['relation'], 
                                         n_latent['relations']['r1']['ss'], 
                                         d_groups)

                # for each entity, we remove it, compute the post-pred across both each
                
                # invert
                new_old = {v : k for k, v in d_groups['d1'].iteritems()}
                # group individually 
                rpp = {}
                rpp_map = {}
                rmap_times = []
                for ei in range(T1_N):
                    orig_grp = latent['domains']['d1']['assignment'][ei]
                    gi = d_groups['d1'][orig_grp]
                    r.remove_entity_from_group('d1', gi, ei)

                    # order
                    gs = new_old.keys()
                    t1 = time.time()
                    rpp_map[ei]= dict(zip([new_old[g] for g in gs], 
                                          r.post_pred_map('d1', 
                                                          gs, 
                                          ei)))

                    rmap_times.append(time.time()-t1)
                    # put it back
                    r.add_entity_to_group('d1', gi, ei)
                postpred_map[relation_name] = rpp_map
                map_times[relation_name] = rmap_times


                
            print 'cpp time=', np.sum(map_times['cpp'])
            print 'parcpp time=', np.sum(map_times['parcpp'])
