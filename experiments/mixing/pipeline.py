from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os
import time

import irm
import irm.data

import cloud

BUCKET_BASE="srm/experiments/mixing"


EXPERIMENTS = [('connmat0', 'fixed40', 'default10')]

def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))


def dataset_connectivity_matrix_params():
    datasets = {'connmat0' : {'seeds' : range(4), 
                              'side_n' : [5, 10], 
                              'class_n' : [5, 10], 
                              'jitter' : [0.001, 0.01, 0.1]}
                }

    for dataset_name, ds in datasets.iteritems():
        for side_n in ds['side_n']:
            for class_n in ds['class_n']:
                for jitter in ds['jitter']:
                    for seed in ds['seeds']:

                        filename_base = "data.%s.%d.%d.%3.2f.%s" % (dataset_name, side_n, class_n, jitter, seed)
                        
                        yield None, [filename_base + ".data", 
                                     filename_base + ".latent", 
                                     filename_base + ".meta"], seed, possible_side_n, class_n, jitter

@files(dataset_connectivity_matrix_params)
def dataset_connectivity_matrix(infile, (data_filename, latent_filename, 
                                         meta_filename), 
                                seed, possible_side_n, class_n, jitter):

    np.random.seed(seed)

    conn_config = {}


    nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(side_n, 
                                                                         conn_config,
                                                                         JITTER=jitter)
    
                
    conn_and_dist = np.zeros(connectivity.shape, 
                    dtype=[('link', np.uint8), 
                           ('distance', np.float32)])

    for ni, (ci, posi) in enumerate(nodes_with_class):
        for nj, (cj, posj) in enumerate(nodes_with_class):
            conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
            conn_and_dist[ni, nj]['distance'] = d(posi, posj)

            
    meta = {'SIDE_N' : SIDE_N, 
            'seed' : seed, 
            'conn_name' : conn_name, 
            'conn_config' : conn_config, 
            'nodes' : nodes_with_class, 
            'connectivity' : connectivity, 
            'conn_and_dist' : conn_and_dist}

    # now create the latents

    model_name= "LogisticDistance" 

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    # FIXME is the assignment vector ground-truth here? 

    HPS = {'mu_hp' : 1.0, 
           'lambda_hp' : 1.0, 
           'p_min' : 0.1, 
           'p_max' : 0.9}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump(meta, open(meta_filename, 'w'))




def create_init(latent_filename, out_filename_base, 
                INIT_N, init= None):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    
    irm_latents = []

    for c in range(INIT_N):
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
        del latent['relations']['R1']['ss']

        filename = "%s.%02d.init" % (out_filename_base, c)
 
        pickle.dump(latent, open(filename, 'w'))



# def create_inference_ld():
#     INITS = SAMPLER_INITS
#     for x in data_generator():
#         filename = x[1]
#         otherargs = x[2:]
#         for seed in range(INITS):
#             outfilename = "%s.ld.%d.pickle" % (filename, init)
#             yield filename, outfilename, init


    

def master_generator():
    for data_name, init_config_name, kernel_config_name:
        for data in get_dataset(data_name):
            inits  = get_init(data_name, init_config_name):
            kernel = get_kernel_conf(kernel_config_name):

            experiment_filename = "%s-%s-%s.experiment" % (data_filename, init_config_name, kernel_config_name)

            exp = {'data' : data_filename, 
                   'inits' : inits, 
                   'kernel' : kernel}

            pickle.dump(exp, open(experiment_filename, 'w'))


@transform("*.experiment", suffix(".experiment"), ".wait")
def run_exp(exp_def, wait_file):
    pass

@transform(run_exp, suffix('.wait'), '.done'):
def get_results(exp_wait, exp_results):
    pass




        
        
