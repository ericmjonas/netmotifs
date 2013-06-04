from nose.tools import * 
import numpy as np
from matplotlib import pylab


import irm
from irm import util 

def test_slice():
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
