import numpy as np

import util
import gibbs
import relation
import synthdata
import pyximport; 
pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                  reload_support=True)
import fastrelation
import irmio
import time


def benchmark():
    np.random.seed(0)

    T1_N = 200
    T2_N = 200

    T1_C = 20
    T2_C = 20

    t1_assign, t2_assign, data, latent_class_matrix = synthdata.create_T1T2_bb(T1_N, T2_N, T1_C, T2_C)

    config = {'types' : {'t1' : {'hps' : 1.0, 
                                 'N' : T1_N}, 
                         't2' : {'hps' : 1.0, 
                                     'N' : T2_N}}, 
            'relations' : { 'R1' : {'relation' : ('t1', 't2'), 
                                    'model' : 'BetaBernoulli', 
                                    'hps' : {'alpha' : 1.0, 
                                             'beta' : 1.0}}}, 
              'data' : {'R1' : data}}


    irm_model = irmio.model_from_config(config, relation_class=relation.FastRelation)
    t1_obj = irm_model.types['t1']
    t2_obj = irm_model.types['t2']

    SAMPLES_N = 50
    for s in range(SAMPLES_N):
        print s
        t1 = time.time()
        gibbs.gibbs_sample_type(t1_obj)
        gibbs.gibbs_sample_type(t2_obj)
        t2 = time.time()

        print "sample", s, "took", t2-t1, "secs"
        print util.count(t1_obj.get_assignments()).values()
        print util.count(t2_obj.get_assignments()).values()

if __name__ == "__main__":
    benchmark()
