import numpy as np
import cPickle as pickle
import gibbs
import irmio


def default_kernel_config():
    return [('conj_gibbs', {})]

def default_kernel_nonconj_config():
    return [('nonconj_gibbs', {'M' : 10}), 
            ('slice_params', {'width' : 0.5})]
    

def do_inference(irm_model, rng, kernel_config):

    """
    By default we do all domains, all relations. 
    We assume a homogeneous model for the moment. 
    """
    for kernel_name, params in kernel_config:
        if kernel_name == 'conj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type(domain_inf, rng)
        elif kernel_name == 'nonconj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type_nonconj(domain_inf, 
                                                params.get("M", 10), 
                                                rng)
        elif kernel_name == "slice_params":
            for relation_name, relation in irm_model.relations.iteritems():
                relation.apply_comp_kernel("slice_sample", rng, params)

        else:
            raise Exception("Malformed kernel config, unknown kernel %s" % kernel_name)


class Runner(object):
    def __init__(latent, data, kernel_config, seed=0):

        # FIXME add seed

        # create the model
        self.rng = irm.RNG()
        
        self.model = irmio.model_from_config(latent, data)
        self.iters = 0
        
    def get_score():
        self.model.total_score()
        
    def run_iters(N, logger=None):
        """
        Run for N iters, per the kernel config
        """
        for i in range(N):
            do_inference(self.model, self.rng, self.kernel_config)
            self.iters += 1

            if logger:
                logger(self.iters, self.model)

    def get_state(self):
        irmio.get_latent(self.model)
        
        
