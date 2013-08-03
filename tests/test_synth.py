from nose.tools import * 

import numpy as np
from irm import data, models

from matplotlib import pylab

def test_generate_bb_t1t1_full():
    np.random.seed(0)
    N = 100
    # first, from the prior entirely
    d = {'domains' : {'d1' : {'N' : N}}, 
            'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                   'model' : 'BetaBernoulli'}}}
    l = {}

    new_latent, new_data = data.synth.prior_generate(l, d)

    assert_equal(new_data['relations']['R1']['data'].shape, (N, N))
    assert_equal(new_data['relations']['R1']['data'].dtype, np.bool)

    # then set some hyperparameters
    new_latent['relations']['R1']['hps']['alpha'] = 0.05
    new_latent['relations']['R1']['hps']['beta'] = 0.05

    del new_latent['relations']['R1']['ss']
    del new_data['relations']['R1']['data']
    new_latent, new_data = data.synth.prior_generate(new_latent, new_data)
    
    p = np.array([v['p'] for v in new_latent['relations']['R1']['ss'].values()])
    # is most of the mass in the edges? 

    p_l = float(np.sum(p < 0.05))/len(p)

    assert( p_l > 0.35)
    assert(float(np.sum(p > 0.95))/len(p) > 0.35)

    # now check that the data groups were done equivalently
    a = new_latent['domains']['d1']['assignment']
    r1d = new_data['relations']['R1']['data']
    groups = np.unique(a)
    ps = []
    for g1 in groups:
        for g2 in groups:
            data_subset = r1d[a==g1, a==g2].flatten()
            p = float(np.sum(data_subset))/len(data_subset)
            ps.append(p)
    p = np.array(ps)

    p_l = float(np.sum(p < 0.05))/len(p)

    assert( p_l > 0.35)
    assert(float(np.sum(p > 0.95))/len(p) > 0.35)

    # set the assignment vector in the latent
    new_latent['domains']['d1']['assignment'] = np.zeros(N)
    del new_latent['relations']['R1']
    del new_data['relations']['R1']['data']
    new_latent, new_data = data.synth.prior_generate(new_latent, new_data)
    p = new_latent['relations']['R1']['ss'][(0, 0)]['p']
    assert_almost_equal(p, float(np.sum(new_data['relations']['R1']['data'].flatten()))/(N*N), 2)

    # 

def test_generate_bb_t1t2_full():
    np.random.seed(0)
    D1_N = 100
    D2_N = 200
    # first, from the prior entirely
    d = {'domains' : {'d1' : {'N' : D1_N}, 
                      'd2' : {'N' : D2_N}},
            'relations' : {'R1' : {'relation' : ('d1', 'd2'), 
                                   'model' : 'BetaBernoulli'}}}
    l = {}

    new_latent, new_data = data.synth.prior_generate(l, d)

    assert_equal(new_data['relations']['R1']['data'].shape, (D1_N, D2_N))
    assert_equal(new_data['relations']['R1']['data'].dtype, np.bool)

    # then set some hyperparameters
    new_latent['relations']['R1']['hps']['alpha'] = 0.05
    new_latent['relations']['R1']['hps']['beta'] = 0.05

    del new_latent['relations']['R1']['ss']
    del new_data['relations']['R1']['data']
    new_latent, new_data = data.synth.prior_generate(new_latent, new_data)
    
    p = np.array([v['p'] for v in new_latent['relations']['R1']['ss'].values()])
    # is most of the mass in the edges? 
    p_l = float(np.sum(p < 0.05))/len(p)

    assert( p_l > 0.35)
    assert(float(np.sum(p > 0.95))/len(p) > 0.35)

    # now check that the data groups were done equivalently
    a1 = new_latent['domains']['d1']['assignment']
    a2 = new_latent['domains']['d2']['assignment']
    r1d = new_data['relations']['R1']['data']

    ps = []
    for g1 in np.unique(a1):
        for g2 in np.unique(a2):
            data_subset = r1d[a1==g1][:, a2==g2].flatten()
            p = float(np.sum(data_subset))/len(data_subset)
            ps.append(p)
    p = np.array(ps)

    p_l = float(np.sum(p < 0.05))/len(p)
    assert( p_l > 0.35)
    assert(float(np.sum(p > 0.95))/len(p) > 0.35)

    # set the assignment vector in the latent
    new_latent['domains']['d1']['assignment'] = np.zeros(D1_N)
    new_latent['domains']['d2']['assignment'] = np.zeros(D2_N)
    del new_latent['relations']['R1']
    del new_data['relations']['R1']['data']
    new_latent, new_data = data.synth.prior_generate(new_latent, new_data)
    p = new_latent['relations']['R1']['ss'][(0, 0)]['p']
    assert_almost_equal(p, float(np.sum(new_data['relations']['R1']['data'].flatten()))/(D1_N*D2_N), 2)

    # 
            

def test_generate_logistic_t1t2_full():
    np.random.seed(1)
    D1_N = 400
    D2_N = 300
    # first, from the prior entirely
    d = {'domains' : {'d1' : {'N' : D1_N}, 
                      'd2' : {'N' : D2_N}},
            'relations' : {'R1' : {'relation' : ('d1', 'd2'), 
                                   'model' : 'LogisticDistance'}}}
    l = {}

    new_latent, new_data = data.synth.prior_generate(l, d)

    assert_equal(new_data['relations']['R1']['data'].shape, (D1_N, D2_N))
    assert_equal(new_data['relations']['R1']['data'].dtype, models.LogisticDistance().data_dtype())

    # then set some hyperparameters
    new_latent['relations']['R1']['hps']['lambda_hp'] = 1.0
    new_latent['relations']['R1']['hps']['mu_hp'] = 15.0
    new_latent['domains']['d1']['assignment'] = np.arange(D1_N) % 400 
    new_latent['domains']['d2']['assignment'] = np.arange(D2_N) % 300
    del new_latent['relations']['R1']['ss']
    del new_data['relations']['R1']['data']
    new_latent, new_data = data.synth.prior_generate(new_latent, new_data)
    
    mu = np.array([v['mu'] for v in new_latent['relations']['R1']['ss'].values()])
    lamb = np.array([v['lambda'] for v in new_latent['relations']['R1']['ss'].values()])
    print "len(mu)=", len(mu)
    np.testing.assert_approx_equal(np.mean(mu), 15.0, 2)
    np.testing.assert_approx_equal(np.mean(lamb), 1.0, 2)
    

# # FIXME test CRP hyperparam
