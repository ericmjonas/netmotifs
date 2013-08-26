import numpy as np
from irm import util

def grid_gibbs(set_func, get_score, vals_list):
    scores = []
    for v in vals_list:
        set_func(v)
        scores.append(get_score())
    i = util.sample_from_scores(scores)
    set_func(vals_list[i])


    
DEFAULT_RELATION_GRIDS = {}
DEFAULT_RELATION_GRIDS['BetaBernoulli'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 2.0, 5.0], [0.1, 0.5, 1.0, 2.0, 5.0]])]
DEFAULT_RELATION_GRIDS['BetaBernoulliNonConj'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 2.0, 5.0], [0.1, 0.5, 1.0, 2.0, 5.0]])]
DEFAULT_RELATION_GRIDS['LogisticDistance'] = [{'mu' : a, 'lamba' : a} for a in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]]
DEFAULT_RELATION_GRIDS['GammaPoisson'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]])]

def default_grid_crp():
    return np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

def default_grid_relation_hps():
    return DEFAULT_RELATION_GRIDS
