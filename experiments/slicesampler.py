import numpy as np
from matplotlib import pylab

"""
Test exponential walk-out slice sampler
and regular (default) slice sampler

Both from neal

"""


def create_interval(f, x_0, y, w, m):
    U = np.random.uniform(0, 1)
    L = x_0 - w * U
    R = L + w
    V = np.random.uniform(0, 1)
    J = np.floor(m * V)
    K = (m - 1) - J 
    
    while J > 0 and y < f(L):
        L -= w
        J -= 1

    while K > 0 and y < f(R):
        R += w
        K -= 1

    return L, R

def create_interval_double(f, x_0, y, w, p):
    U = np.random.uniform(0, 1)
    L = x_0  - w* U
    R = L + w
    K = p

    while K > 0 and (y < f(L) or y < f(R)):
        V = np.random.uniform(0, 1)
        if V < 0.5:
            L -= (R - L)
        else:
            R += (R - L)
        K -= 1
    return L, R

def double_accept(f, x_0, x_1, y, w, L, R):
    Lhat = L
    Rhat = R
    D = False
    while (Rhat - Lhat ) > 1.1*w:
        M = (Lhat + Rhat)/2.
        if (x_0 < M and x_1 >= M) or (x_0 >= M and x_1 < M):
            D = True
        if x_1 < M:
            Rhat = M
        else:
            Lhat = M
        if D and y >= f(Lhat) and y >= f(Rhat):
            return False
    return True
        

def shrink((L, R), f, x_0, y, w, useaccept = False):
    Lbar = L
    Rbar = R
    
    while True:
        U = np.random.uniform(0, 1)
        x1 = Lbar + U * (Rbar - Lbar)

        if useaccept :
            if y < f(x1) and double_accept(f, x_0, x1, y, w, L, R):
                return x1
        else:
            
            if y < f(x1) :
                return x1

        if x1 < x_0:
            Lbar = x1
        else:
            Rbar = x1

def slice_sample(f, x_0, w, m=100):
    
    y = f(x_0) - np.random.exponential(1)

    L, R = create_interval(f, x_0, y, w, m)
    
    return shrink((L, R), f, x_0, y, w)

def slice_sample_double(f, x_0, w, p = 20):
    
    y = f(x_0) - np.random.exponential(1)
    L, R = create_interval_double(f, x_0, y, w, p)
    
    return shrink((L, R), f, x_0, y, w, useaccept=True)
    
def log_norm_dens(x, mu, var):
    c = -np.log(np.sqrt(var*2*np.pi) )
    v = -(x-mu)**2 / (2*var)
    return c + v 

def dens(x):
    D = 5
    return np.logaddexp(log_norm_dens(x, -D, 1.0),  log_norm_dens(x, D, 1.0))

    
ITERS = 10000

vals = np.zeros(ITERS)
x = 0
for i in range(ITERS):
    x = slice_sample(dens, x, 100.)
    vals[i] = x
    print i
xn = np.linspace(-20, 20, 1000)

pylab.plot(xn, np.exp(dens(xn)))
pylab.hist(vals, normed=True, bins=100)

pylab.show()
