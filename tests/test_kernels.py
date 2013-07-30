from nose.tools import * 
import numpy as np
from matplotlib import pylab


import irm
from irm import util 
from irm import models
from irm import model


def test_slice_normal():
    def dens(x): 
        #mixture of gaussian
        mus = [-1.5, 2]
        vars = [1, 1]
        pis = [0.25, 0.75]
        return np.logaddexp.accumulate([(np.log(pi)  +  util.log_norm_dens(x, mu, var)) for (pi, mu, var) in zip(pis, mus, vars)])[-1]
        # return util.log_norm_dens(x, 0, 1.0)

    rng = irm.RNG()
    ITERS = 100000
    
    x = 0
    results = np.zeros(ITERS)
    
    for i in range(ITERS):
        x = irm.slice_sample(x, dens, rng, 0.5)
        results[i] = x
    MIN = -5
    MAX = 5
    BINS = 100
    x = np.linspace(MIN, MAX, BINS)
    bin_width = x[1] - x[0]

    y = [dens(a + bin_width/2) for a in x[:-1]]
    p = np.exp(y)
    p = p/np.sum(p)/(x[1]-x[0])


    hist, bin_edges = np.histogram(results, x, normed=True)

    kl=  util.kl(hist, p)
    assert kl < 0.1
    #pylab.scatter(x[:-1]+ bin_width/2, hist)
    #pylab.plot(x[:-1], p)
    #pylab.show()


def test_slice_exp():
    """
    Test on a distribution with support on the positive reals
    """

    def dens(x): 
        #mixture of gaussian
        lamb = 2.47
        if x < 0:
            return -np.inf
        else:
            return -x * lamb
        
        # return util.log_norm_dens(x, 0, 1.0)

    rng = irm.RNG()
    ITERS = 1000000
    
    x = 0
    results = np.zeros(ITERS)
    
    for i in range(ITERS):
        x = irm.slice_sample(x, dens, rng, 0.5)
        results[i] = x
    MIN = -1
    MAX = 4
    BINS = 101
    x = np.linspace(MIN, MAX, BINS)
    bin_width = x[1] - x[0]

    y = [dens(a + bin_width/2) for a in x[:-1]]
    p = np.exp(y)
    p = p/np.sum(p)/(x[1]-x[0])


    hist, bin_edges = np.histogram(results, x, normed=True)

    kl=  util.kl(hist, p)
    assert kl < 0.1
    # pylab.scatter(x[:-1]+ bin_width/2, hist)
    # pylab.plot(x[:-1], p)
    # pylab.show()


def test_slice_nonconj():
    T1_N = 10
    T2_N = 20
    np.random.seed(0)
    rng = irm.RNG()

    data = np.random.rand(T1_N, T2_N) > 0.5
    data.shape = T1_N, T2_N

    m =  models.BetaBernoulliNonConj()
    r = irm.Relation([('T1', T1_N), ('T2', T2_N)], 
                     data,m)
    hps = m.create_hps()
    hps['alpha'] = 1.0
    hps['beta'] = 1.0

    r.set_hps(hps)

    tf_1 = model.DomainInterface(T1_N, {'r': ('T1', r)})
    tf_1.set_hps({'alpha' : 1.0})
    tf_2 = model.DomainInterface(T2_N, {'r' : ('T2', r)})
    tf_2.set_hps({'alpha' : 1.0})

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

    # build list of coords / heads/tails
    coord_data = {}
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
            # coords = ((tf_1.get_relation_groupid(0, t1_g), 
            #            tf_2.get_relation_groupid(0, t2_g)))
            coord_data[(t1_g, t2_g)] = (heads, tails)

    # get all the components from this relation
    # now the histograms

    for alpha, beta in [(1.0, 1.0), (10.0, 1.0), 
                        (1.0, 10.0),(0.1, 5.0)]:
        coords_hist = {k : [] for k in coord_data}

        print "alpha=", alpha, "beta=", beta, "="*50
        hps['alpha'] = alpha
        hps['beta'] = beta

        r.set_hps(hps)

        ITERS = 100000
        for i in range(ITERS):
            r.apply_comp_kernel("slice_sample", rng, {'width' : 0.2})

            component_data = model.get_components_in_relation([(tf_1, 0), 
                                                            (tf_2, 0)], 
                                                              r)

            for c in coord_data:
                coords_hist[c].append(component_data[c]['p'])
        for c in coords_hist:
            heads, tails = coord_data[c]
            empirical_p = np.mean(coords_hist[c])
            true_map_p = float(heads + alpha) / (heads +tails + alpha + beta)
            print empirical_p - true_map_p
            np.testing.assert_approx_equal(empirical_p, true_map_p, 3)
        # assert_equal(len(np.unique(tf_1.get_assignments())), 10)

