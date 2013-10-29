
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
    # ('drosophila.gp', 'fixed_10_100', 'nc_10'), 
    # ('drosophila.bb', 'fixed_10_100', 'nc_10'), 
    # ('drosophila.ld', 'fixed_10_100', 'nc_10'), 
    # ('drosophila.edp', 'fixed_10_100', 'nc_10'), 
    ('drosophila.gp', 'fixed_100_200', 'anneal_slow_400'), 
    ('drosophila.bb', 'fixed_100_200', 'anneal_slow_400'), 
    ('drosophila.ld', 'fixed_100_200', 'anneal_slow_400'), 
    ('drosophila.edp', 'fixed_100_200', 'anneal_slow_400'), 
    #('drosophila.gp', 'fixed_100_200', 'anneal_glacial_1000'), 
    #('drosophila.bb', 'fixed_100_200', 'anneal_glacial_1000'), 
    #('drosophila.ld', 'fixed_100_200', 'anneal_glacial_1000'), 
    #('drosophila.edp', 'fixed_100_200', 'anneal_glacial_1000'), 
]

INIT_CONFIGS = {'fixed_10_100' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}}
                

slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300
slow_anneal[0][1]['subkernels'][-1][1]['grids']['ExponentialDistancePoisson'] = irm.gridgibbshps.default_grid_exponential_distance_poisson(10.0, 10.0, 50.0)



glacial_anneal = irm.runner.default_kernel_anneal(64.0, 800)

default_nonconj = irm.runner.default_kernel_nonconj_config()

KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},
                  'anneal_glacial_1000' : {'ITERS' : 1000, 
                                           'kernels' : glacial_anneal},
                  'nc_10' : {'ITERS' : 10, 
                             'kernels' : default_nonconj},

                  }

pickle.dump(default_nonconj, open("kernel.config", 'w'))


@files(['synapses.pickle', 'celldata.pickle'],  'countmatrix.pickle')
def create_count_matrix((synapse_infile, celldata_infile), outfile):
    
    data = pickle.load(open(synapse_infile, 'r'))
    synapse_df = data['synapses']

    celldata_df = pickle.load(open(celldata_infile, 'r'))['celldata']

    celldata_df = celldata_df[np.isfinite(celldata_df['post.x'])] # identify missing entities

    cell_ids = celldata_df.index.values

    synapse_df = synapse_df[synapse_df['post.id'].isin(cell_ids) & synapse_df['pre.id'].isin(cell_ids)]
    
    
    # create the matrix
    name_to_pos = {k:v for v, k in enumerate(cell_ids)}
    CELL_N = len(cell_ids)
    conn = np.zeros((CELL_N, CELL_N), dtype=np.int32)
    for rowi, row in synapse_df.iterrows():
        pre_idx = name_to_pos[row['pre.id']]
        post_idx = name_to_pos[row['post.id']]
        conn[pre_idx, post_idx] +=1

    assert len(conn) == len(cell_ids)

    pickle.dump({'cell_ids' : cell_ids, 
                 'conn' : conn}, 
                open(outfile, 'w'))

@files(['synapses.pickle', 'celldata.pickle'], 'distcountmatrix.pickle')
def create_dist_count_matrix((synapse_infile, celldata_infile), outfile):
    """
    The LOCATION of a cell is the mean of its syanptic inputs
    Note this also removes NaNs from the post
    """
    data = pickle.load(open(synapse_infile, 'r'))
    celldata_df = pickle.load(open(celldata_infile, 'r'))['celldata']
    
    synapse_df = data['synapses']
    cell_ids = np.array(data['cell_ids'])


    celldata_df = celldata_df[np.isfinite(celldata_df['post.x'])] # identify missing entities

    cell_ids = celldata_df.index.values

    valid_synapse_df = synapse_df[synapse_df['post.id'].isin(cell_ids) & synapse_df['pre.id'].isin(cell_ids)]
    


    # create the matrix
    name_to_pos = {k:v for v, k in enumerate(cell_ids)}
    CELL_N = len(cell_ids)

    link = np.zeros((CELL_N, CELL_N), dtype=np.int32)
    for rowi, row in valid_synapse_df.iterrows():
        pre_idx = name_to_pos[row['pre.id']]
        post_idx = name_to_pos[row['post.id']]
        link[pre_idx, post_idx] +=1

    # distance between all cells postsynaptically
    dist_post = np.zeros((CELL_N, CELL_N), dtype=np.float32)
    for pre_i, pre  in enumerate(cell_ids):
        for post_i, post in enumerate(cell_ids):
            pre_cell = celldata_df.ix[pre]
            post_cell = celldata_df.ix[post]

            for p, m in [('post', dist_post)]:
                s = 0
                for coord in ['x', 'y', 'z']:
                    field = "%s.%s" % (p, coord)
                    s += (pre_cell[field] - post_cell[field])**2
                d = np.sqrt(s)
                m[pre_i, post_i] = d
    
                        
    conn_data = np.zeros(link.shape, dtype=[('link', np.int32), 
                                            ('distance', np.float32)])
    conn_data['link'] = link
    conn_data['distance'] = dist_post
    assert len(conn_data) == len(cell_ids)

    pickle.dump({'cell_ids' : cell_ids, 
                 'conn' : conn_data}, 
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

@files(create_dist_count_matrix, 
       [td("drosophila" + x) for x in [".ld.data", ".ld.latent", ".ld.meta"]])
def create_latents_ld(infile,
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['conn']
    conn_data = np.zeros(conn.shape, dtype=[('link', np.uint8), 
                                            ('distance', np.float32)])

    conn_data['distance'] = conn['distance']
    conn_data['link'] = conn['link'] > 0 

    model_name= "LogisticDistance"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_data, model_name)

    HPS = {'mu_hp' : 5.0, 
           'lambda_hp' : 5.0, 
           'p_min' : 0.01, 
           'p_max' : 0.95}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, 
                open(meta_filename, 'w'))
    
@files(create_dist_count_matrix, 
       [td("drosophila" + x) for x in [".edp.data", ".edp.latent", ".edp.meta"]])
def create_latents_edp(infile,
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['conn']
    conn_data = np.zeros(conn.shape, dtype=[('link', np.int32), 
                                            ('distance', np.float32)])

    conn_data['distance'] = conn['distance']
    conn_data['link'] = conn['link'] 

    model_name= "ExponentialDistancePoisson"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_data, model_name)

    HPS = {'mu_hp' : 5.0, 
           'rate_scale_hp' : 2.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, 
                open(meta_filename, 'w'))
    

@follows(create_latents_ld)
@follows(create_latents_gp)
@follows(create_latents_bb)
@follows(create_latents_edp)
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

    if 'count_infile' in meta:
        meta_infile = meta['count_infile']
    else:
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

def mat_to_scatter_pts(m):
    """
    Take in a matrix and return
    x_pos
    y_pos
    vals
    """
    ROWS, COLS = m.shape
    x = []
    y = []
    v = []
    for r in range(ROWS):
        for c in range(COLS):
            if m[r, c] > 0:
                x.append(r)
                y.append(c)
                v.append(m[r,c])
    return np.array(x), np.array(y), np.array(v)

@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, ".%d.latent.pickle" % d, 
             ".%d.clusters.pdf" % d)  for d in range(3)])
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
    cell_ids = meta['cell_ids']

    cell_properties = pickle.load(open("celldata.pickle", 'r'))['celldata']
    cell_properties = cell_properties[cell_properties.index.isin(cell_ids)]

    print "CONN_MATRIX.SHAPE", conn_matrix.shape
    print "len(cell_properties)", len(cell_properties)

    t = cell_properties['type']
    type_order = np.unique(t)
    t_to_i = {k : v for v, k in enumerate(type_order)}
    type_ints = cell_properties['type'].apply(lambda x : t_to_i[x])
    
    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    from matplotlib.backends.backend_pdf import PdfPages

    # get data
    
    for chain_pos, (latent_fname, latent_pickle, cluster_filename) in enumerate(out_filenames):
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
        if  data_dict['relations']['R1']['model'] == "BetaBernoulli":
            ax.imshow(conn_sorted > 0, interpolation='nearest', 
                      cmap=pylab.cm.Greys)
        elif data_dict['relations']['R1']['model'] == "GammaPoisson": 
            ax.imshow(np.log(conn_sorted +1), interpolation='nearest', 
                      cmap=pylab.cm.Greys)
        elif data_dict['relations']['R1']['model'] == "LogisticDistance": 
            scatter_x, scatter_y, scatter_v = mat_to_scatter_pts(conn_sorted['link'])

            ax.scatter(scatter_x, scatter_y, s=scatter_v, 
                       c='k', edgecolor='none', 
                       alpha=0.5)
        elif data_dict['relations']['R1']['model'] == "ExponentialDistancePoisson": 
            scatter_x, scatter_y, scatter_v = mat_to_scatter_pts(conn_sorted['link'])
            ax.scatter(scatter_x, scatter_y, s=scatter_v, 
                       c='k', edgecolor='none', 
                       alpha=0.5)

        x_line_offset = 0.5
        y_line_offset = 0.4
        for i in  np.argwhere(np.diff(a[ai]) > 0):
            ax.axhline(i + y_line_offset, c='k', alpha=0.7, linewidth=1.0)
            ax.axvline(i + x_line_offset, c='k', alpha=0.7, linewidth=1.0)
        ax.set_xlim(0, len(conn_matrix))
        ax.set_ylim(len(conn_matrix), 0)
        ax.set_aspect(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('postsynaptic')
        ax.set_ylabel('presynaptic')
        f.tight_layout()
        f.savefig(pp, format='pdf')
        
        pp.close()


        f = pylab.figure(figsize=(20, 15))
        print "true_class_labels=", type_order
        irm.plot.plot_purity_hists_h(f, a, type_ints, 
                                     true_class_labels=type_order)

        f.savefig(cluster_filename, bbox_inches='tight')


        pickle.dump(sample_latent, open(latent_pickle, 'w'))

        
@transform(get_results, suffix(".samples"), [".hypers.pdf"])
def plot_hypers(exp_results, (plot_hypers_filename,)):
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))

    f = pylab.figure(figsize= (12, 8))

    
    chains = [c for c in chains if type(c['scores']) != int]

    irm.experiments.plot_chains_hypers(f, chains, data)

    f.savefig(plot_hypers_filename)

if __name__ == "__main__":
    pipeline_run([create_count_matrix, create_dist_count_matrix, 
                  create_latents_gp, create_latents_ld, create_latents_bb,
                  run_exp, 
                  create_inits, 
                  get_results, plot_scores_z, 
                  plot_hypers, 
                  plot_best_latent
              ], multiprocess=2)
