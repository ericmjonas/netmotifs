from ruffus import *
import numpy as np

import irm
import irm.data
from matplotlib import pylab

def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

SIDE_N = 10

nodes_with_class, connectivity = irm.data.generate.two_class_neighbors(SIDE_N, PROX=2.0, JITTER=0.5, CONN_PROB = 0.7) 

model_name= "LogisticDistance" 
kc = irm.runner.default_kernel_nonconj_config()
kc[0][1]['M'] = 30

# remove slice sampler
#kc.pop()

# create the data for the distance-based model

data = np.zeros(connectivity.shape, 
                dtype=[('link', np.uint8), 
                       ('distance', np.float32)])

for ni, (ci, posi) in enumerate(nodes_with_class):
    for nj, (cj, posj) in enumerate(nodes_with_class):
        data[ni, nj]['link'] = connectivity[ni, nj]
        data[ni, nj]['distance'] = d(posi, posj)
        
irm_config = irm.irmio.default_graph_init(data, model_name)

irm_config['relations']['R1']['hps'] = {'mu_hp' : 1.0, 
                                        'lambda_hp' : 1.0, 
                                        'p_min' : 0.1, 
                                        'p_max' : 0.9}
                                        #'force_mu' : 3.0, 
                                        #'force_lambda' :0.3}
rng = irm.RNG()
model = irm.irmio.model_from_config(irm_config, init='crp', 
                                    rng=rng)

rel = model.relations['R1']
doms = [(model.domains['t1'], 0), (model.domains['t1'], 0)]
for i in range(1000):
    print "iteration", i
    irm.runner.do_inference(model, rng, kc)
    a = model.domains['t1'].get_assignments()
    s = np.argsort(a)
    print s
    print "there are", len(np.unique(a)), "unique classes"
    c_s = connectivity[s]
    c_s = c_s[:, s]

    components = irm.model.get_components_in_relation(doms, rel)
    print components
    pylab.imshow(c_s, interpolation='nearest', cmap=pylab.cm.binary)
    pylab.savefig('test.%04d.png' % i)
    
