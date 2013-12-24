from nose.tools import *
import irm
import numpy as np
from matplotlib import pylab
from irm import runner, irmio
import copy



def test_fixed_k():
    # create synthetic data with K groups
    seed = 0

    np.random.seed(seed)

    GROUP_N = 30
    ENTITIES_PER_GROUP = 10
    N = GROUP_N * ENTITIES_PER_GROUP

    model_name = "BetaBernoulli"
    a = np.random.permutation(np.arange(N) % GROUP_N)
    latent = {'domains' : 
              {'d1' : 
            {'assignment' : a, 
         }}, 
              'relations' : {'R1' : {'hps' : {'alpha' : 0.5, 
                                              'beta' : 0.5}}}}

    # ss = {}
    # for g1 in range(GROUP_N):
    #     for g2 in range(GROUP_N):
    #         ss[(g1, g2)] = {'p' : np.random.beta(0.5, 0.5)}
            
    # latent['domains']['d1']['ss'] = ss

    data = {'domains' : {'d1' : {'N' : N}}, 
            'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                   'model' : model_name}}}
    
    new_latent, new_data = irm.data.synth.prior_generate(latent, data)
    # print new_data
    # m = new_data['relations']['R1']['data']
    # f = pylab.figure()
    # ax = f.add_subplot(1, 2, 1)
    # ax.imshow(m, interpolation='nearest')
    # ax2 = f.add_subplot(1, 2, 2)
    # ai = np.argsort(a).flatten()
    # m2 = m[ai, :]
    # m2 = m2[:, ai]
    # ax2.imshow(m2, interpolation='nearest')
    # pylab.show()

    # create model and initialize with that K
    
    # score

    # do inference


    # does score, assignment vector get better? 

    kernel_config = irm.runner.default_kernel_fixed_config()

    run_truth = runner.Runner(new_latent, new_data, kernel_config, seed=0, 
                              fixed_k=True)

    irmio.estimate_suffstats(run_truth.model, run_truth.rng)

    # get ground truth
    ground_truth_score = run_truth.get_score()

    cleaned_up_latent = run_truth.get_state()

    rand_init = copy.deepcopy(cleaned_up_latent)
    # random init -- just the discrete variables
    # for the time being
    
    for di in cleaned_up_latent['domains']:
        d_N = len(rand_init['domains'][di]['assignment'])
        rand_init['domains'][di]['assignment'] =  np.random.permutation(np.arange(d_N) % GROUP_N)

    for ri  in cleaned_up_latent['relations']:
        del rand_init['relations'][ri]['ss']

    run_actual = runner.Runner(rand_init, new_data, kernel_config, seed=seed)

    rand_init_score = run_actual.get_score()
    print "rand_init_score=", rand_init_score
    print "ground_truth_score=", ground_truth_score
    assert_greater(ground_truth_score, rand_init_score )
    
    iter_count = 0
    ITER_OVER = 1000
    ITERS_TO_RUN = 1

    while (run_actual.get_score() - ground_truth_score) < -50: # well this is sort of bullshit
        run_actual.run_iters(ITERS_TO_RUN)
        iter_count += ITERS_TO_RUN
        print iter_count, model_name, run_actual.get_score(), ground_truth_score
        
        assert_less(iter_count, ITERS_TO_RUN*ITER_OVER, "Too many iterations to get good score")

