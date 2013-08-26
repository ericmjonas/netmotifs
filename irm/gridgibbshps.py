import numpy as np
from irm import util


def component_hypers_grid(relation, grid):
    def set_func(v):
        relation.set_hps(v)
    
    def get_score():
        return relation.total_score()
    
    return grid_gibbs(set_func, get_score, grid)



def grid_gibbs(set_func, get_score, vals_list):
    scores = []
    for v in vals_list:
        set_func(v)
        scores.append(get_score())
    i = util.sample_from_scores(scores)
    set_func(vals_list[i])


    
DEFAULT_RELATION_GRIDS = {}
DEFAULT_RELATION_GRIDS['BetaBernoulli'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 5.0], [0.1, 0.5, 1.0, 5.0]])]
DEFAULT_RELATION_GRIDS['BetaBernoulliNonConj'] =  [{'alpha' : a, 'beta' : b} for a, b in util.cart_prod([[0.1, 0.5, 1.0, 5.0], [0.1, 0.5, 1.0, 5.0]])]

def default_grid_logistic_distance(scale):
    pass

def default_grid_crp():
    return np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

