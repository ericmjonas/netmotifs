from nose.tools import *
import numpy as np
import irm
from matplotlib import pylab


def test_crp_hps():
    vals = [0.1, 1.0, 7.0]
    N = 100
    ITERS = 20
    SAMPLES = 50
    grid = np.logspace(np.log10(0.01), np.log10(40), 100)

    kernel_config = irm.runner.default_kernel_nonconj_config()
    kernel_config = irm.runner.add_domain_hp_grid_kernel(kernel_config, grid)
    # create fake data with crp val
    for alpha in vals:
        latent = {'domains' : 
                  {'d1' : 
                   {'hps' : {'alpha' : alpha}}}}
        model_name = "BetaBernoulli"
        data = {'domains' : {'d1' : {'N' : N}}, 
                'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                       'model' : model_name}}}


        new_latent, new_data = irm.data.synth.prior_generate(latent, data)
        new_latent['domains']['d1']['hps']['alpha'] = 10.0

        run_truth = irm.runner.Runner(new_latent, new_data, kernel_config)
        sample_alphas = []
        for si in range(SAMPLES):
            run_truth.run_iters(ITERS)
            s =run_truth.get_state()
            sample_alphas.append(s['domains']['d1']['hps']['alpha'])
        print sample_alphas
        print "The mean is", np.mean(sample_alphas)

def test_bb_hps():
    vals = [{'alpha' : 1.0, 'beta' : 1.0}, 
            {'alpha' : 0.1, 'beta' : 0.1}, 
            {'alpha' : 1.0, 'beta' : 0.1}, 
            {'alpha' : 0.1, 'beta' : 1.0}, 
            {'alpha' : 1.0, 'beta' : 5.0}]

            
    N = 100
    ITERS = 20
    SAMPLES = 50
    kernel_config = irm.runner.default_kernel_nonconj_config()
    kernel_config = irm.runner.add_relation_hp_grid_kernel(kernel_config)

    fid = open('test.out', 'w')

    # create fake data with crp val
    for hps in vals:
        latent = {'domains' : 
                  {'d1' : 
                   {'hps' : {'alpha' : 10.0}}}, 
                  'relations' : 
                  {'R1' : 
                   {'hps' : hps}}}
                  
        model_name = "BetaBernoulli"

        data = {'domains' : {'d1' : {'N' : N}}, 
                'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                       'model' : model_name}}}

        
        new_latent, new_data = irm.data.synth.prior_generate(latent, data)
        new_latent['relations']['R1']['hps'] = {'alpha' : 1.0, 
                                                'beta' : 1.0}

        run_truth = irm.runner.Runner(new_latent, new_data, kernel_config)
        sample_hps = []
        for si in range(SAMPLES):
            run_truth.run_iters(ITERS)
            s =run_truth.get_state()
            sample_hps.append(s['relations']['R1']['hps'])
        
        print sample_hps
        fid.write("hps=%s\n" % str(hps))
        fid.write("samps= %s\n" % str(sample_hps))


def test_ld_hps():
    vals = [{'mu_hp' : 0.1, 'lambda_hp' : 0.1, 
             'p_min' : 0.01, 'p_max' : 0.80}, 
            {'mu_hp' : 5.0, 'lambda_hp' : 5.0, 
             'p_min' : 0.001, 'p_max' : 0.90}]
            
    N = 500
    ITERS = 20
    SAMPLES = 10
    kernel_config = irm.runner.default_kernel_nonconj_config()
    kernel_config = irm.runner.add_relation_hp_grid_kernel(kernel_config)

    fid = open('ld.test.out', 'w')

    # create fake data with crp val
    for hps in vals:
        latent = {'domains' : 
                  {'d1' : 
                   {'hps' : {'alpha' : 20.0}}}, 
                  'relations' : 
                  {'R1' : 
                   {'hps' : hps}}}
                  
        model_name = "LogisticDistance"

        data = {'domains' : {'d1' : {'N' : N}}, 
                'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                       'model' : model_name}}}

        
        new_latent, new_data = irm.data.synth.prior_generate(latent, data)
        new_latent['relations']['R1']['hps'] = {'lambda_hp' : 1.0, 
                                                'mu_hp' : 1.0, 
                                                'p_min' : 0.01, 
                                                'p_max' : 0.95}

        run_truth = irm.runner.Runner(new_latent, new_data, kernel_config)
        sample_hps = []
        for si in range(SAMPLES):
            run_truth.run_iters(ITERS)
            s =run_truth.get_state()
            sample_hps.append(s['relations']['R1']['hps'])
        
        print sample_hps
        fid.write("hps=%s\n" % str(hps))
        fid.write("samps= %s\n" % str(sample_hps))
