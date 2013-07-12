from matplotlib import pylab

import numpy as np

def logistic(x, mu, lamb):
    return 1.0/(1 + np.exp((x - mu)/lamb))

def parametric_plot():
    """
    Plot the example curves
    """
    MAXX = 100.0
    x = np.linspace(0, MAXX, 1000)

    pylab.figure()
    for mu in np.linspace(0, 100, 8):
        lamb = 4.0
        pylab.plot(x, logistic(x, mu, lamb), linewidth=3.0)
        #pylab.axvline(mu, linestyle='--')
    pylab.grid(1)
    pylab.axis([0, MAXX, 0, 1])


    pylab.figure()
    for c, mu in [('r', 0.0), ('b', 50.0)]:

        for lamb in [0.1, 1.0, 4.0, 10.0, 20, 50]:
            pylab.plot(x, logistic(x, mu, lamb), linewidth=3, 
                       c=c)
        #pylab.axvline(mu, linestyle='--')
    pylab.grid(1)
    pylab.axis([0, MAXX, 0, 1])


    pylab.show()


def prior_sample():
    MAXX = 10.0
    x = np.linspace(0, MAXX, 1000)

    mu_hp = 1.0
    lamb_hp = 2.0
    for i in range(10):
        mu = np.random.exponential(mu_hp)
        lamb = np.random.exponential(lamb_hp)
        pylab.plot(x, logistic(x, mu, lamb), c='k', alpha=0.5, linewidth=2)
    pylab.show()


def single_plot():
    MAXX = 5.0
    x = np.linspace(0, MAXX, 1000)

    mu = 0.7
    lamb = 0.3
    pmin = 0.1
    pmax = 0.9
    
    y = logistic(x, mu, lamb)
    scaled_y = y * (pmax - pmin) + pmin
    
    pylab.subplot(1, 2, 1)
    pylab.plot(x, y, linewidth=3.0)
    pylab.grid(1)
    pylab.axis([0, MAXX, 0, 1])
    pylab.subplot(1, 2, 2)
    pylab.plot(x, scaled_y, linewidth=3.0)
    pylab.grid(1)

    pylab.axis([0, MAXX, 0, 1])
    pylab.show()

#parametric_plot()
#prior_sample()


single_plot()