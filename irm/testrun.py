import sys
import cPickle as pickle
import numpy as np

import irm
import relation
import irmio
import gibbs
import util

RELATION_CLASS = relation.FastRelation

np.random.seed(0)

filename = sys.argv[1]

d = pickle.load(open(filename, 'r'))
data = d['nodes']
connectivity = d['connectivity']
NODEN = connectivity.shape[0]

    
config = {'types' : {'t1' : {'hps' : 1.0, 
                             'N' : NODEN}}, 
          'relations' : { 'R1' : {'relation' : ('t1', 't1'), 
                                  'model' : 'BetaBernoulli', 
                                  'hps' : {'alpha' : 1.0, 
                                           'beta' : 1.0}}}, 
          'data' : {'R1' : connectivity}}

irm_model = irmio.model_from_config(config, relation_class = RELATION_CLASS, 
                                    init='crp')

t1_name = sorted(irm_model.types.keys())[0]
t1_obj = irm_model.types[t1_name]

print "INIT SCORE", irm_model.total_score()

true_class = data['class']
print "COMPUTING TRUE SCORE", "-"*40
#irmio.init_domain(t1_obj, true_class)
#print "TRUE SCORE", irm_model.total_score()

#irmio.init_domain(t1_obj, np.ones(t1_obj.entity_count()))
#print "ALL ONE SCORE", irm_model.total_score()


SAMPLES = 10
ITERS_PER_SAMPLE = 1
for s in range(SAMPLES):
    print "sample", s
    for i in range(ITERS_PER_SAMPLE):
        print "sample", s, "iter", i
        model.do_inference(irm_model)
        print util.count(t1_obj.get_assignments())
    print "score =", irm_model.total_score()
    
