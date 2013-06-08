from matplotlib import pylab

import numpy as np

def logistic(x, pos, scale):
    return 1.0/(1 + np.exp((x - pos)/scale))

MAXX = 100.0
x = np.linspace(0, MAXX, 1000)

pylab.figure()
for pos in np.linspace(0, 100, 8):
    scale = 4.0
    pylab.plot(x, logistic(x, pos, scale), linewidth=3.0)
    #pylab.axvline(pos, linestyle='--')
pylab.grid(1)
pylab.axis([0, MAXX, 0, 1])


pylab.figure()
for c, pos in [('r', 0.0), ('b', 50.0)]:

    for scale in [0.1, 1.0, 4.0, 10.0, 20, 50]:
        pylab.plot(x, logistic(x, pos, scale), linewidth=3, 
                   c=c)
    #pylab.axvline(pos, linestyle='--')
pylab.grid(1)
pylab.axis([0, MAXX, 0, 1])


pylab.show()
