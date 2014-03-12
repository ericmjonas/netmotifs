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
    space_vals =  np.logspace(-1.5, 1.8, 10)*scale
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

def default_grid_logistic_distance_fixed_lambda(scale=1.0):
    space_vals =  np.logspace(-1.5, 1.8, 50)*scale
    p_mins = np.array([0.001, 0.01, 0.02])
    p_maxs = np.array([0.90, 0.80, 0.50, 0.20])
    alpha_betas = [0.1, 1.0, 2.0]
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for alpha_beta in alpha_betas:
                res.append({'lambda' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 
                            'p_scale_alpha_hp' : alpha_beta, 
                            'p_scale_beta_hp' : alpha_beta})
    return res

DEFAULT_RELATION_GRIDS['LogisticDistanceFixedLambda'] = default_grid_logistic_distance_fixed_lambda()

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


def default_grid_square_distance_bump(max_dist, param_weight=0.5, p_min=0.001):
    """
    Note this is the one that EXPLICITLY requries a scale parameter (max_distance)
    to be set!
    """

    p_alphas = np.array([0.1, 1.0, 2.0, 5.0])
    p_betas = np.array([0.1, 1.0, 2.0])
    mu_hps = np.array([max_dist/10., max_dist/5.0, max_dist/3.0])
    hps = []
    for a in p_alphas:
        for b in p_betas:
            for mu_hp in mu_hps:
                hps.append({'p_alpha' : a, 'p_beta': b, 
                       'mu_hp' : mu_hp, 
                       'p_min' : p_min, 
                       'param_weight' : param_weight, 
                       'param_max_distance' : max_dist})

    return hps
DEFAULT_RELATION_GRIDS['SquareDistanceBump'] = default_grid_square_distance_bump(4.0)
# FIXME linear distance


def default_grid_crp():
    return np.logspace(np.log10(1.0), np.log10(100), 40)

def default_grid_relation_hps():
    return DEFAULT_RELATION_GRIDS

def default_grid_exponential_distance_poisson(dist_scale = 1.0, rate_scale_scale = 1.0, 
                                              GRIDN = 10):
    mu = util.logspace(0.1, 1.0, GRIDN) * dist_scale
    rate_scale = util.logspace(0.1, 1.0, GRIDN) * rate_scale_scale
    hps = []
    for m in mu:
        for r in rate_scale:
            hps.append({'mu_hp' : m, 
                        'rate_scale_hp' : r})
    return hps

DEFAULT_RELATION_GRIDS['ExponentialDistancePoisson'] = default_grid_exponential_distance_poisson()


def default_grid_logistic_distance_poisson(dist_scale = 1.0, rate_scale_scale = 1.0, 
                                           GRIDN = 10):
    mu = util.logspace(0.1, 1.0, GRIDN) * dist_scale
    rate_scale = util.logspace(0.1, 1.0, GRIDN) * rate_scale_scale
    hps = []
    for m in mu:
        for r in rate_scale:
            hps.append({'mu_hp' : m, 
                        'lambda' : m, 
                        'rate_scale_hp' : r, 
                        'rate_min' : 0.01})
    return hps

DEFAULT_RELATION_GRIDS['LogisticDistancePoisson'] = default_grid_logistic_distance_poisson()



def default_grid_normal_inverse_chi_sq(mu_scale = 1.0, 
                                       var_scale = 1.0, 
                                       GRIDN = 10):
    mu = np.linspace(-1.0, 1.0, GRIDN+1) * mu_scale #+1 to always include zero
    sigmasq = util.logspace(0.1, 1.0, GRIDN) * var_scale
    kappa = util.logspace(0.1, 10.0, GRIDN)
    nu = util.logspace(0.1, 10.0, GRIDN)
    
    hps = []
    for m in mu:
        for s in sigmasq:
            for k in kappa:
                for n in nu:
                    hps.append({'mu' : m, 
                                'kappa' : k, 
                                'sigmasq' : s, 
                                'nu' : n})
    return hps

DEFAULT_RELATION_GRIDS['NormalInverseChiSq'] = default_grid_normal_inverse_chi_sq()
