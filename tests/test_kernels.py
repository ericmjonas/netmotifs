from nose.tools import * 
import numpy as np
from matplotlib import pylab


import irm
from irm import util 

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
    print "Done" 
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
    print "Done" 
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
    data = np.random.rand(T1_N, T2_N) > 0.5
    data.shape = T1_N, T2_N

    m =  models.BetaBernoulliConj()
    r = relation.Relation([('T1', T1_N), ('T2', T2_N)], 
                     data,m)
    hps = m.create_hps()
    hps['alpha'] = 1.0
    hps['beta'] = 1.0

    r.set_hps(hps)

    tf_1 = model.DomainInterface(T1_N, [('T1', r)])
    tf_1.set_hps(1.0)
    tf_2 = model.DomainInterface(T2_N, [('T2', r)])
    tf_2.set_hps(1.0)

    ### All one group for everyone
    t1_g1 = tf_1.create_group(rng)
    for i in range(T1_N):
        tf_1.add_entity_to_group(t1_g1, i)

    t2_g1 = tf_2.create_group(rng)
    for i in range(T2_N):
        tf_2.add_entity_to_group(t2_g1, i)

    
    # ITERS = 10
    # for i in range(ITERS):
    #     gibbs.gibbs_sample_type(tf_1, rng)
    # assert_equal(len(np.unique(tf_1.get_assignments())), 10)
