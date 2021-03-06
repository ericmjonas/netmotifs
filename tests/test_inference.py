from nose.tools import * 
import numpy as np
import irm
from irm import runner, irmio
import copy

"""
Inference testing of small-sized data, but across all types
and relation classes. 

"""

ITERS_TO_RUN = 1
MODELS =  [
    "BetaBernoulliNonConj", 
           'LogisticDistance', 
           'LogisticDistanceFixedLambda', 
           'LinearDistance', 
           'SigmoidDistance', 
           'GammaPoisson',
           'BetaBernoulli', 
           'NormalDistanceFixedWidth', 

           'ExponentialDistancePoisson',
           'LogisticDistancePoisson', 
    'NormalInverseChiSq',
    #'SquareDistanceBump',  # NOTE THIS IS SO NOT READY FOR PRODUCTION the delta prob is broken
    "MixtureModelDistribution"
]



KERNELS = {'default_nonconj' : irm.runner.default_kernel_nonconj_config(), 
           #'default_anneal':  irm.runner.default_kernel_anneal()
}

INITS = [None, 'sequential']

def test_t1_t1_a(): # the _a is just for filtering
    np.random.seed(0)

    GROUP_N = 5
    ENTITIES_PER_GROUP = 10
    N = GROUP_N * ENTITIES_PER_GROUP


    latent = {'domains' : 
              {'d1' : 
            {'assignment' : np.random.permutation(np.arange(N) % GROUP_N)}, 
           }}

    for model_name in MODELS:
        for kernel_name, kernel_config in KERNELS.iteritems():
            for init in INITS:
                np.random.seed(0)

                data = {'domains' : {'d1' : {'N' : N}}, 
                        'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                               'model' : model_name}}}
                yield check_score_progress, model_name, latent, data, np.random.randint(0, 10000), kernel_config, init 


def test_t1_t1_t1_t1():
    np.random.seed(0)

    GROUP_N = 5
    ENTITIES_PER_GROUP = 10
    N = GROUP_N * ENTITIES_PER_GROUP

    latent = {'domains' : 
              {'d1' : 
            {'assignment' : np.random.permutation(np.arange(N) % GROUP_N)}, 
           }}

    for model_name in MODELS:
        for kernel_name, kernel_config in KERNELS.iteritems():
            for init in INITS:
                np.random.seed(0)

                data = {'domains' : {'d1' : {'N' : N}}, 
                        'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                               'model' : model_name}, 
                                       'R2' : {'relation' : ('d1', 'd1'), 
                                               'model' : model_name}}}

                yield check_score_progress, model_name, latent, data, np.random.randint(0, 10000), kernel_config, init
    
def test_t1_t2():
    np.random.seed(0)

    T1_GRP_N = 5
    T1_EPG = 10

    T2_GRP_N = 14
    T2_EPG = 7

    T1_N = T1_GRP_N * T1_EPG
    T2_N = T2_GRP_N * T2_EPG

    kernel_config = irm.runner.default_kernel_nonconj_config()


    latent = {'domains' : 
              {'d1' : 
               {'assignment' : np.random.permutation(np.arange(T1_N) % T1_GRP_N)}, 
               'd2' : 
               {'assignment' : np.random.permutation(np.arange(T2_N) % T2_GRP_N)}}}

    for model_name in MODELS:
        for kernel_name, kernel_config in KERNELS.iteritems():
            for init in INITS:
                np.random.seed(0)

                data = {'domains' : {'d1' : {'N' : T1_N}, 
                                     'd2' : {'N' : T2_N}}, 
                        'relations' : {'R1' : {'relation' : ('d1', 'd2'), 
                                               'model' : model_name}}}
                yield check_score_progress, model_name, latent, data, np.random.randint(0, 10000), kernel_config, init

def test_t1_t2_t3():
    np.random.seed(1)

    T1_GRP_N = 5
    T1_EPG = 10

    T2_GRP_N = 7
    T2_EPG = 11

    T3_GRP_N = 6
    T3_EPG = 13

    T1_N = T1_GRP_N * T1_EPG
    T2_N = T2_GRP_N * T2_EPG
    T3_N = T3_GRP_N * T3_EPG

    kernel_config = irm.runner.default_kernel_nonconj_config()
    latent = {'domains' : 
              {'d1' : 
               {'assignment' : np.random.permutation(np.arange(T1_N) % T1_GRP_N)}, 
               'd2' : 
               {'assignment' : np.random.permutation(np.arange(T2_N) % T2_GRP_N)}, 
               'd3' : 
               {'assignment' : np.random.permutation(np.arange(T3_N) % T3_GRP_N)}}}
               

    for model_name in MODELS:
        for kernel_name, kernel_config in KERNELS.iteritems():
            for init in INITS:
                np.random.seed(0)

                data = {'domains' : {'d1' : {'N' : T1_N}, 
                                     'd2' : {'N' : T2_N}, 
                                     'd3' : {'N' : T3_N}}, 
                        'relations' : {'R1' : {'relation' : ('d1', 'd2'), 
                                               'model' : model_name}, 
                                       'R2' : {'relation' : ('d2', 'd3'),
                                               'model' : model_name}}}

                yield check_score_progress, model_name, latent, data, np.random.randint(0, 10000), kernel_config, init
    
def check_score_progress(model_name, latent, data, seed, kernel_config, init_type = None, ITERS_TO_RUN=ITERS_TO_RUN):
    print "Running", model_name, "*"*40, np.random.randint(0, 10000)

    np.random.seed(seed)
    new_latent, new_data = irm.data.synth.prior_generate(latent, data)
    # estimate suffstats from the data

    run_truth = runner.Runner(new_latent, new_data, kernel_config, seed=0)
    print "latent=", new_latent

    irmio.estimate_suffstats(run_truth.model, run_truth.rng)
    print "estimated ss" 

    # get ground truth
    ground_truth_score = run_truth.get_score()

    cleaned_up_latent = run_truth.get_state()

    rand_init = copy.deepcopy(cleaned_up_latent)
    # random init -- just the discrete variables
    # for the time being
    
    for di in cleaned_up_latent['domains']:
        d_N = len(rand_init['domains'][di]['assignment'])
        rand_init['domains'][di]['assignment'] = irm.util.crp_draw(d_N, 4.0)

    for ri  in cleaned_up_latent['relations']:
        del rand_init['relations'][ri]['ss']

    run_actual = runner.Runner(rand_init, new_data, kernel_config, seed=seed)

    rand_init_score = run_actual.get_score()
    print "rand_init_score=", rand_init_score
    print "ground_truth_score=", ground_truth_score
    assert_greater(ground_truth_score, rand_init_score )
    
    if init_type != None:
        run_actual.init(init_type) 

    iter_count = 0
    ITER_OVER = 10000
    while (run_actual.get_score() - ground_truth_score) < -50: # well this is sort of bullshit
        run_actual.run_iters(ITERS_TO_RUN)
        iter_count += ITERS_TO_RUN
        print iter_count, model_name, run_actual.get_score(), ground_truth_score
        
        assert_less(iter_count, ITERS_TO_RUN*ITER_OVER, "Too many iterations to get good score")

    
# def test_t1_t1_anneal():
#     np.random.seed(0)

#     GROUP_N = 5
#     ENTITIES_PER_GROUP = 10
#     N = GROUP_N * ENTITIES_PER_GROUP
#     kernel_config = irm.runner.default_kernel_anneal()
#     latent = {'domains' : 
#               {'d1' : 
#             {'assignment' : np.random.permutation(np.arange(N) % GROUP_N)}, 
#            }}

#     for model_name in MODELS:
        
#         data = {'domains' : {'d1' : {'N' : N}}, 
#                 'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
#                                        'model' : model_name}}}
#         yield check_score_progress, model_name, latent, data, np.random.randint(0, 10000), kernel_config, 100

