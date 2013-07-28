import numpy as np
from matplotlib import pylab

N = 500
CLASSN = 7

true_assign = np.arange(N) % CLASSN


tv = true_assign.argsort()
tv_i = true_assign[tv]

a = [tv_i]

true_vect= np.arange(N) % CLASSN + 10
t2 = (np.arange(N) + 10) % CLASSN # correct assignment, wrong order
t3 = (np.arange(N) + 10) % (CLASSN+1)
for v in [true_vect, t2, t3]:
    tv = v.argsort()
    a.append(true_assign[tv])

i = np.vstack(a)

f = pylab.figure()
ax = f.add_subplot(1, 1, 1)

ax.imshow(i, interpolation='nearest')
ax.set_aspect(10.0)
pylab.show()
