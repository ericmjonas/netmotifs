import numpy as np
import cPickle as pickle
import time
import gibbs
import irmio
import kernels, models, model
import pyirm
import sys

def default_kernel_config():
    return [('conj_gibbs', {})]

def default_kernel_nonconj_config():
    return [('nonconj_gibbs', {'M' : 10}), 
            ('slice_params', {'width' : 0.5})]
    
def do_inference(irm_model, rng, kernel_config, reverse=False, 
                 states_at_temps = None):

    """
    By default we do all domains, all relations. 
    We assume a homogeneous model for the moment. 
    """
    step = 1
    res = None
    if reverse:
        step = -1
    for kernel_name, params in kernel_config[::step]:
        t1 = time.time()
        if kernel_name == 'conj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type(domain_inf, rng, params.get("impotent", False))
        elif kernel_name == 'nonconj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type_nonconj(domain_inf, 
                                                params.get("M", 10), 
                                                rng, 
                                                params.get("impotent", False))
        elif kernel_name == "slice_params":
            for relation_name, relation in irm_model.relations.iteritems():
                relation.apply_comp_kernel("slice_sample", rng, params)
        elif kernel_name == "tempered_transitions":
            temps = params['temps']
            subkernels = params['subkernels']
            
            kernels.tempered_transitions(irm_model, rng, temps, 
                                         irmio.get_latent, 
                                         lambda x, y : irmio.set_model_latent(x, y, rng), 
                                         model.IRM.set_temp, 
                                         lambda x, y, r: do_inference(x, y, subkernels, r))
        elif kernel_name == "parallel_tempering":
            temps = params['temps']
            subkernels = params['subkernels']
            if len(states_at_temps) != len(temps):
                raise Exception("Insufficient latent states")
            states_at_temps = kernels.parallel_tempering(irm_model, states_at_temps, 
                                                         rng, temps, 
                                                         irmio.get_latent, 
                                                         lambda x, y : irmio.set_model_latent(x, y, rng), 
                                                         model.IRM.set_temp, 
                                                         lambda x, y: do_inference(x, y, subkernels))

            irmio.set_model_latent(irm_model, states_at_temps[0], rng)
            res = states_at_temps
        else:
            raise Exception("Malformed kernel config, unknown kernel %s" % kernel_name)
        t2 = time.time()
        print "kernels:", kernel_name, "%3.2f sec" % (t2-t1)
        return res

class Runner(object):
    def __init__(self, latent, data, kernel_config, seed=0):

        # FIXME add seed

        # create the model
        self.rng = pyirm.RNG()
        
        self.model = irmio.create_model_from_data(data, rng=self.rng)
        irmio.set_model_latent(self.model, latent, self.rng)
        self.iters = 0
        
        self.kernel_config = kernel_config

        self.PT = False
        if len(kernel_config) == 1 and kernel_config[0][0] == "parallel_tempering":
            self.PT = True
            self.chain_states = []

            # create the chain states
            for t in kernel_config[0][1]['temps']:
                self.chain_states.append(irmio.get_latent(self.model))

    def get_score(self):
        return self.model.total_score()
        
    def run_iters(self, N, logger=None):
        """
        Run for N iters, per the kernel config
        """
        for i in range(N):
            if self.PT :
                self.chain_states = do_inference(self.model, self.rng, self.kernel_config, 
                                                states_at_temps = self.chain_states)
            else:
                do_inference(self.model, self.rng, self.kernel_config)
            self.iters += 1

            if logger:
                logger(self.iters, self.model)

    def get_state(self):
        return irmio.get_latent(self.model)
        

if __name__ == "__main__":
    # command-line runner

    latent_filename = sys.argv[1]
    data_filename = sys.argv[2]
    config_filename = sys.argv[3]
    iters = int(sys.argv[4])
    latent = pickle.load(open(latent_filename))
    data = pickle.load(open(data_filename))
    config = pickle.load(open(config_filename))
    
    run = Runner(latent, data, config)
    def logger(iter, model):
        print iter

    run.run_iters(iters, logger)

