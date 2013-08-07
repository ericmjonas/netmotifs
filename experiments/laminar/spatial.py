import numpy as np
from matplotlib import pylab

N = 2000
x = np.random.rand(N, 2)



pylab.scatter(x[:, 0], x[:, 1])



pylab.show()


