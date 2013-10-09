import numpy as np

class Bernoulli(object):
    dtype = np.bool

    def sample(self, p):
        return np.random.rand() < p


class Poisson(object):
    dtype = np.int32

    def sample(self, p):
        return np.random.poisson(p)

class Normal(object):
    """

    """
    dtype = np.float32

    def sample(self, (mu, var)):
        return np.random.normal(mu, np.sqrt(var))

