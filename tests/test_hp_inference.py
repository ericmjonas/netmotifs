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
