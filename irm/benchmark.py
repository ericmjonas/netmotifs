import numpy as np

import util
import gibbs
import relation
import synthdata

def benchmark():
    import irmio

    T1_N = 100
    T2_N = 100

    T1_C = 10
    T2_C = 10

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

    SAMPLES_N = 5
    for s in range(SAMPLES_N):
        print s
        gibbs.gibbs_sample_type(t1_obj)
        gibbs.gibbs_sample_type(t2_obj)
        print util.count(t1_obj.get_assignments()).values()
        print util.count(t2_obj.get_assignments()).values()

if __name__ == "__main__":
    benchmark()
