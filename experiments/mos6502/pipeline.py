from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob
import time
from matplotlib import pylab
import matplotlib.gridspec as gridspec

import irm
import irm.data

import cloud

BUCKET_BASE="srm/experiments/mos6502"


EXPERIMENTS = [('mos6502.all.bb', 'fixed_20_200', 'default_100'), 
               ('mos6502.all.ld', 'fixed_20_200', 'default_nc_100'), 
               ('mos6502.dir.bb', 'fixed_20_200', 'default_100'), 
               ('mos6502.dir.ld', 'fixed_20_200', 'default_nc_100'), 
               
               # ('mos6502.all.bb', 'fixed_20_200', 'default_100'), 
               # ('mos6502.all.ld', 'fixed_20_200', 'default_nc_100'), 
           ]

INIT_CONFIGS = {'fixed_20_200' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
                'fixed_2_40' : {'N' : 2, 
                                'config' : {'type' : 'fixed', 
                                              'group_num' : 40}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}}
                
WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)
                

default_nonconj = irm.runner.default_kernel_nonconj_config()
default_nonconj[1][1]['width'] = 500. 

default_conj = irm.runner.default_kernel_config()

pickle.dump(default_nonconj, open("test.config", 'w'))

KERNEL_CONFIGS = {'default_nc_100' : {'ITERS' : 100, 
                                  'kernels' : default_nonconj},
                  'default_nc_10' : {'ITERS' : 10, 
                                  'kernels' : default_nonconj},
                  'default_10' : {'ITERS' : 10, 
                                  'kernels' : default_conj},
                  'default_100' : {'ITERS' : 100, 
                                  'kernels' : default_conj},
                  'default_nc_1' : {'ITERS' : 1, 
                                  'kernels' : default_nonconj},
                  'default_1' : {'ITERS' : 1, 
                                  'kernels' : default_conj},
                  }


def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))


@transform('*.adjmat.pickle', regex("(.+)\.adjmat.pickle"), r'%s/mos6502.\1.data.pickle' % WORKING_DIR)
def data_mos6502_adjmat(infile, all_file):
    data = pickle.load(open(infile, 'r'))

    
    pickle.dump({'dist_matrix' : data['adj_mat'], 
                 'infile' : infile}, open(all_file, 'w'))

@transform(data_mos6502_adjmat, suffix(".data.pickle"), [".ld.data", ".ld.latent", ".ld.meta"])
def create_latents_ld(infile, 
                      (data_filename, latent_filename, meta_filename)):
    print "INPUT FILE IS", infile
    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "LogisticDistance" 

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'mu_hp' : 1000., 
           'lambda_hp' : 1000., 
           'p_min' : 0.05, 
           'p_max' : 0.95}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))


@transform(data_mos6502_adjmat, suffix(".data.pickle"), [".bb.data", ".bb.latent", ".bb.meta"])
def create_latents_bb(infile, 
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['dist_matrix']['link']
    
    model_name= "BetaBernoulli"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn, model_name)

    HPS = {'alpha' : 1.0, 
           'beta' : 1.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, open(meta_filename, 'w'))


def create_init(latent_filename, out_filenames, 
                init= None):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    
    irm_latents = []

    for c, out_f in enumerate(out_filenames):
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        if init['type'] == 'fixed':
            group_num = init['group_num']

            a = np.arange(len(latent['domains']['d1']['assignment'])) % group_num
            a = np.random.permutation(a)

        elif init['type'] == 'crp':
            alpha = init['alpha']
        else:
            raise NotImplementedError("Unknown init type")
            
        if c > 0: # first one stays the same
            latent['domains']['d1']['assignment'] = a

        # delete the suffstats
        if 'ss' in latent['relations']['R1']:
            del latent['relations']['R1']['ss']

        pickle.dump(latent, open(out_f, 'w'))


def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s.%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            

@follows(create_latents_ld, create_latents_bb)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    irm.experiments.create_init(latent_filename, data_filename, 
                                out_filenames, 
                                init= init_config['config'])



def inference_run_ld(latent_filename, 
                     data_filename, 
                     kernel_config,  ITERS, seed):

    latent = from_bucket(latent_filename)
    data = from_bucket(data_filename)

    chain_runner = irm.runner.Runner(latent, data, kernel_config, seed)

    scores = []
    times = []
    def logger(iter, model):
        print "Iter", iter
        scores.append(model.total_score())
        times.append(time.time())
    chain_runner.run_iters(ITERS, logger)
        
    return scores, chain_runner.get_state(), times


def experiment_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            inits = ["%s.%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
            exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
            yield [data_filename, inits], exp_name, kernel_config_name

@follows(create_inits)
@files(experiment_generator)
def run_exp((data_filename, inits), wait_file, kernel_config_name):
    # put the filenames in the data
    to_bucket(data_filename)
    [to_bucket(init_f) for init_f in inits]

    kc = KERNEL_CONFIGS[kernel_config_name]
    CHAINS_TO_RUN = len(inits)
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    
    jids = cloud.map(inference_run_ld, inits, 
                     [data_filename]*CHAINS_TO_RUN, 
                     [kernel_config]*CHAINS_TO_RUN,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     _env='connectivitymotif', 
                     _type='f2')

    pickle.dump({'jids' : jids, 
                'data_filename' : data_filename, 
                'inits' : inits, 
                'kernel_config_name' : kernel_config_name}, 
                open(wait_file, 'w'))


@transform(run_exp, suffix('.wait'), '.samples')
def get_results(exp_wait, exp_results):
    
    d = pickle.load(open(exp_wait, 'r'))
    
    chains = []
    # reorg on a per-seed basis
    for chain_data in cloud.iresult(d['jids'], ignore_errors=True):
        
        chains.append({'scores' : chain_data[0], 
                       'state' : chain_data[1], 
                       'times' : chain_data[2]})
        
        
    pickle.dump({'chains' : chains, 
                 'exp' : d}, 
                open(exp_results, 'w'))

# @transform(get_results, suffix(".samples"), [".latent.pdf"])
# def plot_latent(exp_results, (plot_latent_filename, )):
#     sample_d = pickle.load(open(exp_results))
#     chains = sample_d['chains']
    
#     exp = sample_d['exp']
#     data_filename = exp['data_filename']
#     data = pickle.load(open(data_filename))
#     data_basename, _ = os.path.splitext(data_filename)
#     meta = pickle.load(open(data_basename + ".meta"))


#     # nodes_with_class = meta['nodes']
#     # conn_and_dist = meta['conn_and_dist']

#     # true_assignvect = nodes_with_class['class']

#     chains = [c for c in chains if type(c['scores']) != int]
#     CHAINN = len(chains)

#     f = pylab.figure(figsize= (12, 8))
#     ax_purity_control = f.add_subplot(2, 2, 1)
#     ax_z = f.add_subplot(2, 2, 2)
#     ax_score = f.add_subplot(2, 2, 3)
    
#     # ###### plot purity #######################
#     # ###
#     # tv = true_assignvect.argsort()
#     # tv_i = true_assignvect[tv]
#     # vals = [tv_i]
#     # # get the chain order 
#     # chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
#     # sorted_assign_matrix = []
#     # for di in chains_sorted_order: 
#     #     d = chains[di] 
#     #     sample_latent = d['state']
#     #     a = np.array(sample_latent['domains']['d1']['assignment'])
#     #     print "di=%d unique classes:"  % di, np.unique(a)
#     #     sorted_assign_matrix.append(a)
#     # irm.plot.plot_purity(ax_purity_control, true_assignvect, sorted_assign_matrix)

#     ###### zmatrix
#     av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
#     z = irm.util.compute_zmatrix(av)    

#     irm.plot.plot_zmatrix(ax_z, z)

#     ### Plot scores
#     for di, d in enumerate(chains):
#         subsamp = 4
#         s = np.array(d['scores'])[::subsamp]
#         t = np.array(d['times'])[::subsamp] - d['times'][0]
#         ax_score.plot(t, s, alpha=0.7, c='k')

#     ax_score.tick_params(axis='both', which='major', labelsize=6)
#     ax_score.tick_params(axis='both', which='minor', labelsize=6)
#     ax_score.set_xlabel('time (s)')
#     ax_score.grid(1)
    
#     f.tight_layout()

#     f.savefig(plot_latent_filename)

@transform(get_results, suffix(".samples"), [".scoresz.pdf"])
def plot_scores_z(exp_results, (plot_latent_filename,)):
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']

    d = pickle.load(open(meta_infile, 'r'))
    conn = d['dist_matrix']['link']
    orig_data = pickle.load(open(d['infile']))
    print "len(dist_matrix):", conn.shape
    #cell_types = orig_data['types'][:len(conn)]

    # nodes_with_class = meta['nodes']
    # conn_and_dist = meta['conn_and_dist']

    # true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_purity_control = f.add_subplot(2, 2, 1)
    ax_z = f.add_subplot(2, 2, 2)
    ax_score = f.add_subplot(2, 2, 3)
    
    ###### zmatrix
    av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
    z = irm.util.compute_zmatrix(av)    

    z_ord = irm.plot.plot_zmatrix(ax_z, z)

    ### Plot scores
    for di, d in enumerate(chains):
        subsamp = 4
        s = np.array(d['scores'])[::subsamp]
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

    ax_score.tick_params(axis='both', which='major', labelsize=6)
    ax_score.tick_params(axis='both', which='minor', labelsize=6)
    ax_score.set_xlabel('time (s)')
    ax_score.grid(1)
    
    f.tight_layout()

    f.savefig(plot_latent_filename)
    f = pylab.figure(figsize=(20, 4))

    
@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, )  for d in range(2)])
def plot_best_latent(exp_results, 
                     out_filenames):
    print "Plotting best latent", exp_results
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']
    print "meta_infile=", meta_infile

    d = pickle.load(open(meta_infile, 'r'))
    dist_matrix = d['dist_matrix']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)
    print "CHAINN=", CHAINN, "out_filenames=", out_filenames
    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (latent_fname, ) in enumerate(out_filenames):
        print "plotting chain", chain_pos

        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']

        model = data['relations']['R1']['model']
        irm.experiments.plot_latent(sample_latent, dist_matrix, latent_fname, 
                                    model=model, PLOT_MAX_DIST=10000)
    


pipeline_run([data_mos6502_adjmat, 
              create_inits, get_results, plot_best_latent, plot_scores_z])
                        
