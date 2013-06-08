from ruffus import *
import numpy as np

import irm
import irm.data
from matplotlib import pylab

SIDE_N = 10

nodes_with_class, connectivity = irm.data.generate.one_class_neighbors(SIDE_N, PROX=3.0, JITTER=0.5, CONN_PROB = 0.2) 

model_name= "BetaBernoulliNonConj" 
kc = irm.runner.default_kernel_nonconj_config()
irm_config = irm.irmio.default_graph_init(connectivity, model_name)

rng = irm.RNG()
model = irm.irmio.model_from_config(irm_config, init='crp', 
                                    rng=rng)


for i in range(100):
    print "iteration", i
    irm.runner.do_inference(model, rng, kc)
    a = model.domains['t1'].get_assignments()
    s = np.argsort(a)
    c_s = connectivity[s]
    c_s = c_s[:, s]
    pylab.imshow(c_s, interpolation='nearest', cmap=pylab.cm.binary)
    pylab.savefig('test.%04d.png' % i)
    
