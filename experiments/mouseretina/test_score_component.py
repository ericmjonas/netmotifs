import cPickle as pickle
import numpy as np
from irm.util import logistic
from matplotlib import pylab

def log_exp_dist(x, lamb):
    if x < 0:
        return -np.inf
    else:
        return np.log(lamb) + -lamb * x; 

LOOPMAX = 1000

def slice_sample(x, P, w):
    Pstar = P(x)
    uprime = np.log(np.random.rand()) + Pstar
    r = np.random.rand()

    x_l = x - r*w
    x_r = x + (1-r) * w

    loopcount = 0
    while (P(x_l) > uprime) and (loopcount < LOOPMAX):
        x_l -= w
        loopcount += 1
    while (P(x_r) > uprime) and (loopcount < LOOPMAX):
        x_r += w
        loopcount += 1

    if loopcount == LOOPMAX:
        raise Exception("didn't converge")
        
    loopcnt = 0
    while loopcount < LOOPMAX:
        xprime = np.random.uniform(x_l, x_r)
        if P(xprime) > uprime:
            return xprime
        if xprime > x:
            x_r = xprime
        else:
            x_l = xprime
        loopcnt +=1
    raise Exception("Slice sampling failed")


def score_component(conn_true, conn_false, mu, lamb, mu_hp, lamb_hp, 
                    p_min, p_max):
    score = 0.0
    p_range = p_max - p_min 
    
    for d in conn_true:
        p = logistic(d, mu, lamb) * p_range + p_min
        score += np.log(p)

    for d in conn_true:
        p = logistic(d, mu, lamb) * p_range + p_min
        score += np.log(1.0 - p)
        
    prior = log_exp_dist(mu, mu_hp)
    prior += log_exp_dist(lamb, lamb_hp)

    score += prior

    return score

def mh(conn_true, conn_false, mu, lamb, mu_hp, lamb_hp, p_min, p_max, ITERS=10):
    """
    MH of params

    """

    sigma = 1.0
    
    for i in range(ITERS) :
        mu_prime = np.random.normal(mu, sigma)
        s1 = score_component(conn_true, conn_false, mu, lamb, mu_hp, 
                            lamb_hp, p_min, p_max)

        s2 = score_component(conn_true, conn_false, mu_prime, lamb, mu_hp, 
                            lamb_hp, p_min, p_max)
        if np.random.rand() < np.exp(s2 - s1):
            mu = mu_prime
        
        lamb_prime = np.random.normal(lamb, sigma)
        s1 = score_component(conn_true, conn_false, mu, lamb, mu_hp, 
                            lamb_hp, p_min, p_max)

        s2 = score_component(conn_true, conn_false, mu, lamb_prime, mu_hp, 
                            lamb_hp, p_min, p_max)
        if np.random.rand() < np.exp(s2 - s1):
            lamb = lamb_prime
        print i, mu, lamb
    return mu, lamb

def slice(conn_true, conn_false, mu, lamb, mu_hp, lamb_hp, p_min, p_max, ITERS=10):
    """
    MH of params

    """

    for i in range(ITERS) :
        def score_mu(mu):
            return score_component(conn_true, conn_false, mu, lamb, mu_hp, 
                                   lamb_hp, p_min, p_max)
        mu = slice_sample(mu, score_mu, 20.0)

        def score_lamb(mu):
            return score_component(conn_true, conn_false, mu, lamb, mu_hp, 
                                   lamb_hp, p_min, p_max)
        lamb = slice_sample(lamb, score_lamb, 20.0)

    return mu, lamb
        

def grid():

    data = pickle.load(open("../mouseretina/component.0.1.pickle", 'r'))
    conn_true = data['conn_dist_hist']
    conn_false = data['noconn_dist_hist']

    N = 200
    p_max = 0.95
    p_min = 0.05
    x = np.zeros((N, N))
    mus = np.linspace(0.0, 10, N)
    lambs = np.linspace(0.0, 4, N)
    for i, mu in enumerate(mus):
        for j, lamb in enumerate(lambs):
            mu_hp = 10.0
            lamb_hp = 10.0

            x[i, j] = score_component(conn_true, conn_false, mu, lamb, 
                                      mu_hp, lamb_hp, 
                                      p_min, p_max)
    x -= np.max(x)

    mu_i, lamb_i =  np.unravel_index(np.argmax(x), x.shape)

    pylab.figure()
    pylab.plot(data['bins'][:-1], data['p'])
    pylab.plot(data['bins'], logistic(data['bins'], mus[mu_i], 
                                      lambs[lamb_i])*(p_max - p_min) + p_min, c='r')

    pylab.figure()
    pylab.imshow(x)
    pylab.colorbar()
    pylab.figure()
    for ki, k in enumerate([1, 10, 20, 30]):
        pylab.subplot(4, 1, ki+1)
        p = np.exp(x[:, k]) / np.sum(np.exp(x[:, k]))
        pylab.plot(mus, p)
        pylab.grid()

    pylab.show()



def one_axis_test():
    data = pickle.load(open("../mouseretina/component.2.3.pickle", 'r'))
    conn_true = data['conn_dist_hist']
    conn_false = data['noconn_dist_hist']
    print "true_count = ", len(conn_true)
    print "false_count=", len(conn_false)
    p_max = 0.98
    p_min = 0.02
    mu_hp = 2.0
    lamb_hp = 5.0
    mu = 10.0
    lamb=10.0
    pylab.figure()
    pylab.plot(data['bins'][:-1], data['p'])
    for S in range(10):
        mu, lamb = slice(conn_true, conn_false, mu, lamb, 
                      mu_hp, lamb_hp, p_min, p_max, ITERS=30)
        
        pylab.plot(data['fine_bins'], logistic(data['fine_bins'], mu, 
                                          lamb)*(p_max - p_min) + p_min, c='r')

    pylab.show()
one_axis_test()

