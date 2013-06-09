from ruffus import *
import numpy as np

import irm
import irm.data
from matplotlib import pylab

def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

SIDE_N = 8

nodes_with_class, connectivity = irm.data.generate.one_class_neighbors(SIDE_N, PROX=3.0, JITTER=0.5, CONN_PROB = 0.9) 

model_name= "LogisticDistance" 
kc = irm.runner.default_kernel_nonconj_config()
kc[0][1]['M'] = 30
kc.pop()

# create the data for the distance-based model

data = np.zeros((SIDE_N * SIDE_N, SIDE_N * SIDE_N), 
                dtype=[('link', np.uint8), 
                       ('distance', np.float32)])

for ni, (ci, posi) in enumerate(nodes_with_class):
    for nj, (cj, posj) in enumerate(nodes_with_class):
        data[ni, nj]['link'] = connectivity[ni, nj]
        data[ni, nj]['distance'] = d(posi, posj)

irm_config = irm.irmio.default_graph_init(data, model_name)
irm_config['relations']['R1']['hps'] = {'mu_hp' : 1.0, 
                                        'lambda_hp' : 1.0, 
                                        'p_min' : 0.01, 
                                        'p_max' : 0.99}
rng = irm.RNG()
model = irm.irmio.model_from_config(irm_config, init='crp', 
                                    rng=rng)


for i in range(100):
    print "iteration", i
    irm.runner.do_inference(model, rng, kc)
    a = model.domains['t1'].get_assignments()
    s = np.argsort(a)
    print "there are", len(np.unique(s)), "unique classes"
    c_s = connectivity[s]
    c_s = c_s[:, s]
    pylab.imshow(c_s, interpolation='nearest', cmap=pylab.cm.binary)
    pylab.savefig('test.%04d.png' % i)
    
