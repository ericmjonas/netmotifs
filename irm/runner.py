import numpy as np
import cPickle as pickle
import time
import gibbs
import irmio
import kernels, models, model, gridgibbshps
import pyirm
import sys
import copy

def default_kernel_config():
    return [('conj_gibbs', {})]

def kernel_nonconj_contmh_config():
    return [('nonconj_gibbs', {'M' : 10}), 
            ('continuous_mh_params', {'iters' : 10, 'log_scale_min': -3, 'log_scale_max' : 4})] 
    
def default_kernel_nonconj_config():
    return [('nonconj_gibbs', {'M' : 10}), 
            ('slice_params', {'width' : 0.0})] # use default

def default_kernel_fixed_config():
    return [('fixed_gibbs', {}), 
            ('slice_params', {'width' : 0.0})] # use default
    

def add_domain_hp_grid_kernel(kernel_list, grid=None):
    kl = copy.deepcopy(kernel_list)
    if grid == None:
        grid = gridgibbshps.default_grid_crp()
    kl.append(('domain_hp_grid', {'grid': grid}))
    return kl

def add_relation_hp_grid_kernel(kernel_list, grids=None):
    kl = copy.deepcopy(kernel_list)
    if grids == None:
        grids = gridgibbshps.default_grid_relation_hps()
    else:
        default_grid = copy.deepcopy(gridgibbshps.default_grid_relation_hps())
        default_grid.update(grids)
        grids = default_grid

    kl.append(('relation_hp_grid', {'grids': grids}))
    return kl


def default_kernel_anneal(start_temp = 32.0, iterations=100):
    dk = default_kernel_nonconj_config()
    dk = add_domain_hp_grid_kernel(dk)
    dk = add_relation_hp_grid_kernel(dk)
    
    return [('anneal', {'anneal_sched': {'start_temp' : start_temp, 
                                         'stop_temp' : 1.0, 
                                         'iterations' : iterations}, 
                         'subkernels': dk})]

              
def do_inference(irm_model, rng, kernel_config, iteration,
                 reverse=False, 
                 states_at_temps = None):

    """
    By default we do all domains, all relations. 
    We assume a homogeneous model for the moment. 
    
    The way values are returned from PT here is an abomination, and should
    be resolved at some point 
    """
    step = 1
    res = {'kernel_times' : []}

    if reverse:
        step = -1
    for kernel_name, params in kernel_config[::step]:
        t1 = time.time()
        if kernel_name == 'conj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type(domain_inf, rng, 
                                        params.get("impotent", False))
        elif kernel_name == 'fixed_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_fixed_k(domain_inf, rng, 
                                           params.get("impotent", False))
        elif kernel_name == 'nonconj_gibbs':
            for domain_name, domain_inf in irm_model.domains.iteritems():
                gibbs.gibbs_sample_type_nonconj(domain_inf, 
                                                params.get("M", 10), 
                                                rng, 
                                                params.get("impotent", False))
        elif kernel_name == "slice_params":
            for relation_name, relation in irm_model.relations.iteritems():
                relation.apply_comp_kernel("slice_sample", rng, params)
        elif kernel_name == "continuous_mh_params":
            for relation_name, relation in irm_model.relations.iteritems():
                relation.apply_comp_kernel("continuous_mh", rng, params)
        elif kernel_name == "tempered_transitions":
            temps = params['temps']
            subkernels = params['subkernels']
            
            kernels.tempered_transitions(irm_model, rng, temps, 
                                         irmio.get_latent, 
                                         lambda x, y : irmio.set_model_latent(x, y, rng), 
                                         model.IRM.set_temp, 
                                         lambda x, y, r: do_inference(x, y, subkernels, iteration, r))
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
                                                         lambda x, y: do_inference(x, y, subkernels, iteration))

            irmio.set_model_latent(irm_model, states_at_temps[0], rng)
            res = states_at_temps

        elif kernel_name == "anneal":
            temp_sched = params['anneal_sched']
            subkernels = params['subkernels']
            # i know this is gross, I don't care
            

            sub_res = kernels.anneal(irm_model, rng, temp_sched, 
                                     iteration, 
                                     model.IRM.set_temp, 
                                     lambda x, y: do_inference(x, y, subkernels,
                                                               iteration))
            for v in sub_res['kernel_times']:
                res['kernel_times'].append(v)

        elif kernel_name == "domain_hp_grid":
            grid = params['grid']
            kernels.domain_hp_grid(irm_model, rng, grid)
        elif kernel_name == "relation_hp_grid":
            grids = params['grids']
            kernels.relation_hp_grid(irm_model, rng, grids)
            

        else:
            raise Exception("Malformed kernel config, unknown kernel %s" % kernel_name)
        t2 = time.time()
        res['kernel_times'].append((kernel_name, t2-t1))
        print "kernels:", kernel_name, "%3.2f sec" % (t2-t1)
    return res

class Runner(object):
    def __init__(self, latent, data, kernel_config, seed=None, 
                 fixed_k = False):

        # FIXME add seed
        print "FIXED_K=", fixed_k
        # create the model
        self.rng = pyirm.RNG()
        if seed != None:
            pyirm.set_seed(self.rng, seed)
        
        self.model = irmio.create_model_from_data(data, rng=self.rng, 
                                                  fixed_k = fixed_k)
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

    def init(self, init_type):
        if init_type == "sequential": # Fixme we really should propagate params through here someday
            print "RUNNING SEQUENTIAL INIT" 
            kernels.sequential_init(self.model, self.rng)

    def get_score(self):
        return self.model.total_score()
        
    def run_iters(self, N, logger=None):
        """
        Run for N iters, per the kernel config
        """
        for i in range(N):
            res = None
            if self.PT :
                self.chain_states = do_inference(self.model, self.rng,
                                                 self.kernel_config, self.iters,
                                                 states_at_temps = self.chain_states)
            else:
                res = do_inference(self.model, self.rng, self.kernel_config, self.iters)
            self.iters += 1

            if logger:
                logger(self.iters, self.model, res)

    def get_state(self, include_ss=True):
        return irmio.get_latent(self.model, include_ss)
        

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
        print iter, model.get_score()

    run.run_iters(iters, logger)

