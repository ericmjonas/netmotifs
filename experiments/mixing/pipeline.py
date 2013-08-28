from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob
import time
from matplotlib import pylab

import irm
import irm.data

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


import cloud

BUCKET_BASE="srm/experiments/mixing"


EXPERIMENTS = [('trivial', 'fixed_4_10', 'default20'), 
               # ('connmat0', 'fixed_10_40', 'default20'), 
               ('connmat0', 'fixed_10_40', 'default200'), 
               ('connmat0', 'fixed_10_40', 'nc_contmh_200'), 
               ('connmat0', 'fixed_10_40', 'default_anneal'),
               # ('connmat0', 'fixed_10_100', 'default200'), 
               #('connmat0', 'fixed_10_40', 'pt200'),
               # ('connmat0', 'fixed_10_40', 'default20_m100'), 
           ]

INIT_CONFIGS = {'fixed_4_10' : {'N' : 4, 
                             'config' : {'type' : 'fixed', 
                                         'group_num' : 10}}, 
                'fixed_10_40' : {'N' : 10, 
                                'config' : {'type' : 'fixed', 
                                            'group_num' : 40}}, 
                'fixed_10_100' : {'N' : 10, 
                                'config' : {'type' : 'fixed', 
                                            'group_num' : 100}}}
                

default_nonconj = irm.runner.default_kernel_nonconj_config()
nonconj_m100 = copy.deepcopy(default_nonconj)
nonconj_m100[0][1]['M'] = 100
default_anneal = irm.runner.default_kernel_anneal()


KERNEL_CONFIGS = {'default20' : {'ITERS' : 20, 
                                 'kernels' : default_nonconj},
                  'default20_m100' : {'ITERS' : 20, 
                                      'kernels' : nonconj_m100},
                  'default200' : {'ITERS' : 200, 
                                  'kernels' : default_nonconj},
                  'nc_contmh_200' : {'ITERS' : 200, 
                                  'kernels' : irm.runner.kernel_nonconj_contmh_config()},
                  'pt200' : {'ITERS' : 200, 
                             'kernels' : [('parallel_tempering', 
                                           {'temps' : [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], 
                                            'subkernels' : default_nonconj})]}, 
                  'default_anneal' : {'ITERS' : 200, 
                                      'kernels' : default_anneal}}


def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))


def dataset_connectivity_matrix_params():
    datasets = {'connmat0' : {'seeds' : range(2), 
                              'side_n' : [5, 10], 
                              'class_n' : [5, 10], 
                              'nonzero_frac' : [0.1, 0.2, 0.5], 
                              'jitter' : [0.001, 0.01, 0.1], 
                              'models' : ['ld', 'lind']}, 
                'trivial' : {'seeds' : range(1), 
                             'side_n' : [4], 
                             'class_n' : [2], 
                             'nonzero_frac' : [1.0], 
                             'jitter' : [0.001], 
                             'models' : ['ld', 'lind']}}
    
    for dataset_name, ds in datasets.iteritems():
        for side_n in ds['side_n']:
            for class_n in ds['class_n']:
                for nonzero_frac in ds['nonzero_frac']:
                    for jitter in ds['jitter']:

                        for seed in ds['seeds']:
                            for model in ds['models']:

                                filename_base = "data.%s.%s.%d.%d.%3.3f.%3.3f.%s" % (dataset_name, model, side_n, class_n, nonzero_frac, jitter, seed)

                                yield None, [filename_base + ".data", 
                                             filename_base + ".latent", 
                                filename_base + ".meta"], model, seed, side_n, class_n, nonzero_frac, jitter
                            
@files(dataset_connectivity_matrix_params)
def dataset_connectivity_matrix(infile, (data_filename, latent_filename, 
                                         meta_filename), 
                                model, seed, side_n, class_n, nonzero_frac, jitter):

    np.random.seed(seed)
            
    conn_config = {}

    for c1 in range(class_n):
        for c2 in range(class_n):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (np.random.uniform(1.0, 4.0), 
                                         np.random.uniform(0.1, 0.9))
    if len(conn_config) == 0:
        conn_config[(0, 0)] = (np.random.uniform(1.0, 4.0), 
                               np.random.uniform(0.4, 0.9))


    nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(side_n, 
                                                                         conn_config,
                                                                         JITTER=jitter)
    
                
    conn_and_dist = np.zeros(connectivity.shape, 
                    dtype=[('link', np.uint8), 
                           ('distance', np.float32)])

    for ni, (ci, posi) in enumerate(nodes_with_class):
        for nj, (cj, posj) in enumerate(nodes_with_class):
            conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
            conn_and_dist[ni, nj]['distance'] = dist(posi, posj)

            
    meta = {'SIDE_N' : side_n,
            'seed' : seed, 
            'class_n' : class_n, 
            'conn_config' : conn_config, 
            'nodes' : nodes_with_class, 
            'connectivity' : connectivity, 
            'conn_and_dist' : conn_and_dist}

    # now create the latents

    if model == 'ld':
        model_name= "LogisticDistance" 

        HPS = {'mu_hp' : 1.0, 
               'lambda_hp' : 1.0, 
               'p_min' : 0.1, 
               'p_max' : 0.9}
    elif model == 'lind':
        model_name= "LinearDistance"

        HPS = {'mu_hp' : 1.0, 
               'p_alpha' : 1.0, 
               'p_beta': 1.0, 
               'p_min' : 0.01}

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)
    irm_latent['domains']['d1']['assignment'] = nodes_with_class['class']

    # FIXME is the assignment vector ground-truth here? 

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump(meta, open(meta_filename, 'w'))




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



# def create_inference_ld():
#     INITS = SAMPLER_INITS
#     for x in data_generator():
#         filename = x[1]
#         otherargs = x[2:]
#         for seed in range(INITS):
#             outfilename = "%s.ld.%d.pickle" % (filename, init)
#             yield filename, outfilename, init


def get_dataset(data_name):
    return glob.glob("data.%s.*.data" %  data_name)

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s.%d.init" % (name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            
            # inits  = get_init(data_name, init_config_name)
            # kernel = get_kernel_conf(kernel_config_name)

            # experiment_filename = "%s-%s-%s.experiment" % (data_filename, init_config_name, kernel_config_name)

            # exp = {'data' : data_filename, 
            #        'inits' : inits, 
            #        'kernel' : kernel}

            #pickle.dump(exp, open(experiment_filename, 'w'))

@follows(dataset_connectivity_matrix)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    create_init(latent_filename, out_filenames, 
                init= init_config['config'])



def inference_run_ld(latent_filename, 
                     data_filename, 
                     kernel_config,  ITERS, seed):

    latent = from_bucket(latent_filename)
    data = from_bucket(data_filename)

    SAVE_EVERY = 100
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

            inits = ["%s.%d.init" % (name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
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

def parse_filename(fn):
     data_part = fn.split('-')[0]
     s = data_part.split(".")
     print "S=", s
     s.pop(0)
     dataset_name = s[0]
     model = s[1]
     side_n = int(s[2])
     class_n = int(s[3])
     nonzero_frac = float(s[4] + '.' + s[5])
     jitter = float(s[6] + '.' + s[7])
     return {'dataset_name' : dataset_name, 
             'model' : model, 
             'side_n' : side_n, 
             'class_n' : class_n, 
             'nonzero_frac' : nonzero_frac, 
             'jitter' : jitter}


@transform(get_results, suffix(".samples"), [".latent.pdf"])
def plot_latent(exp_results, (plot_latent_filename, )):
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))


    nodes_with_class = meta['nodes']
    conn_and_dist = meta['conn_and_dist']

    true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_purity_control = f.add_subplot(2, 2, 1)
    ax_z = f.add_subplot(2, 2, 2)
    ax_score = f.add_subplot(2, 2, 3)
    
    ###### plot purity #######################
    ###
    tv = true_assignvect.argsort()
    tv_i = true_assignvect[tv]
    vals = [tv_i]
    # get the chain order 
    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    sorted_assign_matrix = []
    for di in chains_sorted_order: 
        d = chains[di] 
        sample_latent = d['state']
        a = np.array(sample_latent['domains']['d1']['assignment'])
        print "di=%d unique classes:"  % di, np.unique(a)
        sorted_assign_matrix.append(a)
    irm.plot.plot_purity(ax_purity_control, true_assignvect, sorted_assign_matrix)

    ###### zmatrix
    av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
    z = irm.util.compute_zmatrix(av)    

    irm.plot.plot_zmatrix(ax_z, z)

    ### Plot scores
    for di, d in enumerate(chains):
        subsamp = 4
        s = np.array(d['scores'])[::subsamp]
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        if di == 0:
            ax_score.plot(t, s, alpha=0.7, c='r', linewidth=3)
        else:
            ax_score.plot(t, s, alpha=0.7, c='k')

    ax_score.tick_params(axis='both', which='major', labelsize=6)
    ax_score.tick_params(axis='both', which='minor', labelsize=6)
    ax_score.set_xlabel('time (s)')
    ax_score.grid(1)
    
    file_params = parse_filename(exp_results)
    f.suptitle(str(file_params))

    f.savefig(plot_latent_filename)
    

pipeline_run([dataset_connectivity_matrix, create_inits, run_exp, 
              get_results, plot_latent], multiprocess=6)
                        
