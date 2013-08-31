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

DEFAULT_RELATION_GRIDS['GammaPoisson'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]])]

def default_grid_logistic_distance(scale=1.0):
    space_vals =  np.logspace(-1.5, 1.8, 50)*scale
    p_mins = np.array([0.001, 0.01, 0.02])
    p_maxs = np.array([0.90, 0.80, 0.50, 0.20])
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for p_max in p_maxs:
                res.append({'lambda_hp' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 'p_max' : p_max})
    return res

DEFAULT_RELATION_GRIDS['LogisticDistance'] = default_grid_logistic_distance()

def default_grid_normal_distance_fixed_width():
    p_mins = np.array([0.001, 0.01, 0.02])
    mus = np.array([0.1, 1.0])
    widths = np.array([0.1, 0.5, 1.0])
    alphabetas = [0.1, 1.0, 2.0]
    hps = []
    for p_min in p_mins:
        for mu in mus:
            for width in widths:
                for a in alphabetas:
                    for b in alphabetas:
                        hps.append({'mu_hp' : mu, 
                                    'p_min' : p_min, 
                                    'width' : width, 
                                    'p_alpha' : a,
                                    'p_beta' : b})
    return hps

DEFAULT_RELATION_GRIDS['NormalDistanceFixedWidth'] = default_grid_normal_distance_fixed_width()

# FIXME linear distance


def default_grid_crp():
    return np.logspace(np.log10(0.1), np.log10(50), 40)

def default_grid_relation_hps():
    return DEFAULT_RELATION_GRIDS
