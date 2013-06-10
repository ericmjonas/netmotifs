import numpy as np
import cPickle as pickle
from matplotlib import pylab

d = pickle.load(open('output.pickle', 'r'))
pylab.figure()
pylab.plot(d['scores'])
pylab.xlabel('iteration')
pylab.ylabel('log score')
pylab.savefig('scores.png')
