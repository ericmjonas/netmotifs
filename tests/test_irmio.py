from nose.tools import * 
import numpy as np
from numpy.testing import assert_approx_equal

from irm import irmio, model

import irm
"""
Tests of the IRM IO 
"""
np.random.seed(0)

latent_simple_nonconj = {'domains' : {'d1' :{ 'hps' : {'alpha': 1.0}, 
                                             'N' : 5, 
                                             'assignment' : [0, 0, 1, 1, 2]}}, 
                        'relations' : {'R1' : {'relation' :  ('d1', 'd1'), 
                                               'model' : "BetaBernoulliNonConj", 
                                               'hps' : {'alpha' : 1.0, 
                                                        'beta' : 1.0}}},
                         'ss' : {'R1' : {(0, 0) : {'p' : 0.0}, 
                                         (0, 1) : {'p' : 0.01}, 
                                         (0, 2) : {'p' : 0.02}, 
                                         (1, 0) : {'p' : 0.10}, 
                                         (1, 1) : {'p' : 0.11}, 
                                         (1, 2) : {'p' : 0.12}, 
                                         (2, 0) : {'p' : 0.20}, 
                                         (2, 1) : {'p' : 0.21}, 
                                         (2, 2) : {'p' : 0.22}}}}

data_simple_nonconj = {'R1' : np.random.rand(5, 5) > 0.5}


def test_simple_nonconj():
    rng = irm.RNG()
    irm_model = irmio.model_from_latent(latent_simple_nonconj, 
                                        data_simple_nonconj, rng=rng)
    
    a = irm_model.domains['d1'].get_assignments()
    axes = irm_model.relations['R1'].get_axes()
    axes_objs = [(irm_model.domains[dn], irm_model.domains[dn].get_relation_pos('R1')) 
                 for dn in axes]

    comps = model.get_components_in_relation(axes_objs,
                                             irm_model.relations['R1'])

    g0 = a[0]
    g1 = a[2]
    g2 = a[4]

    assert_approx_equal(comps[g0, g0]['p'], 0.0)
    assert_approx_equal(comps[g0, g1]['p'], 0.01)
    assert_approx_equal(comps[g0, g2]['p'], 0.02)

    assert_approx_equal(comps[g1, g0]['p'], 0.1)
    assert_approx_equal(comps[g1, g1]['p'], 0.11)
    assert_approx_equal(comps[g1, g2]['p'], 0.12)


    assert_approx_equal(comps[g2, g0]['p'], 0.2)
    assert_approx_equal(comps[g2, g1]['p'], 0.21)
    assert_approx_equal(comps[g2, g2]['p'], 0.22)


def test_simple_nonconj_inout():
    rng = irm.RNG()
    irm_model = irmio.model_from_latent(latent_simple_nonconj, 
                                        data_simple_nonconj, rng=rng)
    latent = irmio.get_latent(irm_model)
    print latent['ss']['R1']
    irmio.latent_equality(latent, latent_simple_nonconj)

def test_runner():
    run = irm.runner.Runner(latent_simple_nonconj, data_simple_nonconj, 
                            irm.runner.default_kernel_nonconj_config())
    run.get_score()
    run.run_iters(10)
    run.get_score()
    run.get_state()
