
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

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


import cloud

BUCKET_BASE="srm/experiments/drosophila"

WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)


EXPERIMENTS = [
    ('drosophila.gp', 'fixed_10_100', 'nc_10'), 
    ('drosophila.bb', 'fixed_10_100', 'nc_10'), 
    ('drosophila.gp', 'fixed_100_200', 'anneal_slow_400'), 
    ('drosophila.bb', 'fixed_100_200', 'anneal_slow_400'), 
]

INIT_CONFIGS = {'fixed_10_100' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}}
                

slow_anneal = irm.runner.default_kernel_anneal()
default_nonconj = irm.runner.default_kernel_nonconj_config()

KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},
                  'nc_10' : {'ITERS' : 10, 
                             'kernels' : default_nonconj},

                  }



@files('synapses.pickle', 'countmatrix.pickle')
def create_count_matrix(infile, outfile):
    
    data = pickle.load(open(infile, 'r'))
    synapse_df = data['synapses']
    cell_ids = data['cell_ids']
    
    
    # create the matrix
    name_to_pos = {k:v for v, k in enumerate(cell_ids)}
    CELL_N = len(cell_ids)
    conn = np.zeros((CELL_N, CELL_N), dtype=np.int32)
    for rowi, row in synapse_df.iterrows():
        pre_idx = name_to_pos[row['pre.id']]
        post_idx = name_to_pos[row['post.id']]
        conn[pre_idx, post_idx] +=1

    pickle.dump({'cell_ids' : cell_ids, 
                 'conn' : conn}, 
                open(outfile, 'w'))

def get_dataset(data_name):
    return glob.glob("%s.data" %  data_name)


def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(td(data_name)):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


@files(create_count_matrix, 
       [td("drosophila" + x) for x in [".gp.data", ".gp.latent", ".gp.meta"]])
def create_latents_gp(infile,
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['conn']
    
    model_name= "GammaPoisson"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn, model_name)

    HPS = {'alpha' : 1.0, 
           'beta' : 1.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, open(meta_filename, 'w'))


@files(create_count_matrix, 
       [td("drosophila" + x) for x in [".bb.data", ".bb.latent", ".bb.meta"]])
def create_latents_bb(infile,
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['conn']
    conn = conn > 0 
    conn = conn.astype(np.uint8)
    
    model_name= "BetaBernoulli"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn, model_name)

    HPS = {'alpha' : 1.0, 
           'beta' : 1.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, open(meta_filename, 'w'))

@follows(create_latents_gp)
@follows(create_latents_bb)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    irm.experiments.create_init(latent_filename, data_filename, 
                                out_filenames, 
                                init= init_config['config'])

def experiment_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(td(data_name)):
            name, _ = os.path.splitext(data_filename)

            inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
            exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
            yield [data_filename, inits], exp_name, kernel_config_name

@follows(create_inits)
@files(experiment_generator)
def run_exp((data_filename, inits), wait_file, kernel_config_name):
    # put the filenames in the data
    print "uploading", data_filename
    irm.experiments.to_bucket(data_filename, BUCKET_BASE)
    [irm.experiments.to_bucket(init_f, BUCKET_BASE) for init_f in inits]

    kc = KERNEL_CONFIGS[kernel_config_name]
    CHAINS_TO_RUN = len(inits)
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    
    jids = cloud.map(irm.experiments.inference_run, inits, 
                     [data_filename]*CHAINS_TO_RUN, 
                     [kernel_config]*CHAINS_TO_RUN,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     [BUCKET_BASE]*CHAINS_TO_RUN, 
                     _label="%s-%s-%s" % (data_filename, inits[0], 
                                          kernel_config_name), 
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
                       'times' : chain_data[2], 
                       'latents' : chain_data[3]})
        
        
    pickle.dump({'chains' : chains, 
                 'exp' : d}, 
                open(exp_results, 'w'))

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


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_z = pylab.subplot2grid((2,2), (0, 0))
    ax_score = pylab.subplot2grid((2,2), (0, 1))
    ax_purity =pylab.subplot2grid((2,2), (1, 0), colspan=2)
    
    ### Plot scores
    for di, d in enumerate(chains):
        subsamp = 4
        s = np.array(d['scores'])[::subsamp]
        print "Scores=", s
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

    f.tight_layout()

    f.savefig(plot_latent_filename)

@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, ".%d.latent.pickle" % d)  for d in range(3)])
def plot_best_latent(exp_results, 
                     out_filenames):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data_dict = pickle.load(open(data_filename, 'r'))
    meta_filename = data_filename[:-4] + "meta"
    m = pickle.load(open(meta_filename, 'r'))
    meta_infile = m['infile']
    meta = pickle.load(open(meta_infile, 'r'))
    conn_matrix = meta['conn']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    from matplotlib.backends.backend_pdf import PdfPages

    # get data
    
    for chain_pos, (latent_fname, latent_pickle) in enumerate(out_filenames):
        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        a =  irm.util.canonicalize_assignment(np.array(sample_latent['domains']['d1']['assignment']))
        ai = np.argsort(a).flatten()
        
        pp = PdfPages(latent_fname)
        
        f = pylab.figure()
        ax = f.add_subplot(1, 1, 1)

        conn_sorted = conn_matrix[ai]
        conn_sorted = conn_sorted[:, ai]
        if  conn_sorted.dtype == np.uint8:
            ax.imshow(conn_sorted > 0, interpolation='nearest', 
                      cmap=pylab.cm.Greys)
        elif conn_sorted.dtype == np.int32:
            ax.imshow(np.log(conn_sorted +1), interpolation='nearest', 
                      cmap=pylab.cm.Greys)

        x_line_offset = 0.5
        y_line_offset = 0.4
        for i in  np.argwhere(np.diff(a[ai]) > 0):
            ax.axhline(i + y_line_offset, c='b', alpha=0.7, linewidth=1.0)
            ax.axvline(i + x_line_offset, c='b', alpha=0.7, linewidth=1.0)

        ax.set_xticks([])
        ax.set_yticks([])

        f.savefig(pp, format='pdf')
        
        pp.close()

        pickle.dump(sample_latent, open(latent_pickle, 'w'))

if __name__ == "__main__":
    pipeline_run([create_count_matrix, create_latents_gp,
                  run_exp, 
                  create_inits, 
                  get_results, plot_scores_z, 
                  plot_best_latent], multiprocess=2)
