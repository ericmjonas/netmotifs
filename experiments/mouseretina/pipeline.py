from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob
import time
from matplotlib import pylab
import matplotlib
import pandas
import colorbrewer

import matplotlib.gridspec as gridspec

import irm
import irm.data
import util
from irm import rand

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


import cloud

BUCKET_BASE="srm/experiments/mouseretina"

WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

EXPERIMENTS = [
    ('retina.1.0.ld.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.1.ld.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.2.ld.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.3.ld.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.0.bb.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.1.bb.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.2.bb.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.1.3.bb.0.0', 'fixed_20_100', 'anneal_slow_400'), 
    ('retina.count.edp', 'fixed_20_100', 'anneal_slow_400'), 
    #('retina.1.0.ld.truth', 'truth_100', 'anneal_slow_400'), 
    ('retina.1.0.ld.0.0', 'fixed_10_20', 'debug'), 
    ('retina.count.edp', 'fixed_10_20', 'debug'), 
    ('retina.1.0.bb.0.0', 'fixed_10_10', 'debug_fixed'), 
               
]

THOLDS = [0.01, 0.1, 0.5, 1.0]
    
MULAMBS = [1.0, 5.0, 10.0, 20.0, 50.0]
PMAXS = [0.95, 0.9, 0.7]

BB_ALPHAS = [1.0]
BB_BETAS = [1.0]


            
INIT_CONFIGS = {'fixed_10_200' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
                'fixed_10_20' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 20}}, 
                'fixed_10_10' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 10}}, 
                'fixed_20_100' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'truth_10' : {'N' : 10, 
                              'config' : {'type' : 'truth'}}, 
                'truth_100' : {'N' : 100, 
                              'config' : {'type' : 'truth'}}, 
}


slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300

def generate_ld_hypers():
    space_vals =  irm.util.logspace(1.0, 80.0, 40)
    p_mins = np.array([0.001, 0.005, 0.01])
    p_maxs = np.array([0.99, 0.95, 0.90, 0.80])
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for p_max in p_maxs:
                res.append({'lambda_hp' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 'p_max' : p_max})
    return res


slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistance'] = generate_ld_hypers()

KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},

                  'debug' : {'ITERS' : 10, 
                             'kernels' : irm.runner.default_kernel_nonconj_config()}, 
                  'debug_fixed' : {'ITERS' : 40, 
                                   'fixed_k' : True, 
                             'kernels' : irm.runner.default_kernel_fixed_config()}, 

                  }


def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


def create_tholds():
    """
    systematicaly vary the threshold for "synapse" and whether or not
    we use the z-axis
    """
    infiles = ['conn.areacount.pickle', 
               'soma.positions.pickle']
    for use_x in [1]:
        for tholdi, thold in enumerate(THOLDS):
            outfile = td("retina.%d.%d.data.pickle" % (use_x, tholdi))
            yield infiles, [outfile], thold, use_x

@files(create_tholds)
def data_retina_adj_bin((conn_areacount_infile, positions_infile), 
                        (retina_outfile,), AREA_THOLD, USE_X):
    """
    From the raw file, create the adjacency matrix. 

    NOTE THAT WE PERMUTE THE ROWS

    Bewcause in the raw file things are already sorted by their 'type'
    and this makes a lot of interpretation challenging
    """

    data = pickle.load(open(conn_areacount_infile, 'r'))
    area_mat = data['area_mat']['area']
    positions_data = pickle.load(open(positions_infile, 'r'))
    pos_vec = positions_data['pos_vec']
    NEURON_N = 950 # only the ones for which we also have position data

    np.random.seed(0)
    cell_id_permutation = np.random.permutation(NEURON_N)


    area_mat_sub = area_mat[:NEURON_N, :NEURON_N]

    area_mat_sub = area_mat_sub[cell_id_permutation, :]
    area_mat_sub = area_mat_sub[:, cell_id_permutation]
    pos_vec = pos_vec[cell_id_permutation]

    dist_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

    cell_types = data['types'][cell_id_permutation]

    dist_matrix['link'] = area_mat_sub > AREA_THOLD
    for n1 in range(NEURON_N):
        for n2 in range(NEURON_N):
            p1 = pos_vec[n1]
            p2 = pos_vec[n2]
            if not USE_X:
                p1[0] = 0
                p2[0] = 0
            
            dist_matrix[n1, n2]['distance'] = dist(p1, p2)

    pickle.dump({'dist_matrix' : dist_matrix, 
                 'area_thold' : AREA_THOLD, 
                 'types' : cell_types, 
                 'cell_id_permutation' : cell_id_permutation,
                 'infile' : conn_areacount_infile}, open(retina_outfile, 'w'))


def create_tholds():
    """
    systematicaly vary the threshold for "synapse" and whether or not
    we use the z-axis
    """
    infiles = ['conn.areacount.pickle', 
               'soma.positions.pickle']
    for use_x in [1]:
        for tholdi, thold in enumerate(THOLDS):
            outfile = td("retina.%d.%d.data.pickle" % (use_x, tholdi))
            yield infiles, [outfile], thold, use_x

@files(['conn.areacount.pickle', 
        'soma.positions.pickle'], ('data/retina.count.data.pickle', ))
def data_retina_adj_count((conn_areacount_infile, positions_infile), 
                          (retina_outfile,)):
    """
    From the raw file, create the adjacency matrix. 

    NOTE THAT WE PERMUTE THE ROWS

    Bewcause in the raw file things are already sorted by their 'type'
    and this makes a lot of interpretation challenging
    """

    data = pickle.load(open(conn_areacount_infile, 'r'))
    area_mat = data['area_mat']['count']
    positions_data = pickle.load(open(positions_infile, 'r'))
    pos_vec = positions_data['pos_vec']
    NEURON_N = 950 # only the ones for which we also have position data

    np.random.seed(0)
    cell_id_permutation = np.random.permutation(NEURON_N)


    area_mat_sub = area_mat[:NEURON_N, :NEURON_N]

    area_mat_sub = area_mat_sub[cell_id_permutation, :]
    area_mat_sub = area_mat_sub[:, cell_id_permutation]
    pos_vec = pos_vec[cell_id_permutation]

    dist_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.int32), 
                                  ('distance', np.float32)])

    cell_types = data['types'][cell_id_permutation]

    dist_matrix['link'] = area_mat_sub 
    for n1 in range(NEURON_N):
        for n2 in range(NEURON_N):
            p1 = pos_vec[n1]
            p2 = pos_vec[n2]
            
            dist_matrix[n1, n2]['distance'] = dist(p1, p2)

    pickle.dump({'dist_matrix' : dist_matrix, 
                 'types' : cell_types, 
                 'cell_id_permutation' : cell_id_permutation,
                 'infile' : conn_areacount_infile}, open(retina_outfile, 'w'))
                
def create_latents_ld_params():
    for a in create_tholds():
        inf = a[1][0]
        for mli, mulamb in enumerate(MULAMBS):
            for pi, p in enumerate(PMAXS):
                outf_base = inf[:-len('.data.pickle')]
                outf = "%s.ld.%d.%d" % (outf_base, mli, pi)
                yield inf, [outf + '.data', 
                            outf + '.latent', outf + '.meta'], mulamb, p
        
@follows(data_retina_adj_bin)
@files(create_latents_ld_params)
def create_latents_ld(infile, 
                      (data_filename, latent_filename, meta_filename), 
                      mulamb, pmax):

    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "LogisticDistance" 

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'mu_hp' : mulamb,
           'lambda_hp' : mulamb,
           'p_min' : 0.05, 
           'p_max' : pmax}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))

def create_latents_bb_params():
    for a in create_tholds():
        inf = a[1][0]
        for ai, alpha in enumerate(BB_ALPHAS):
            for bi, beta in enumerate(BB_BETAS):
                outf_base = inf[:-len('.data.pickle')]
                outf = "%s.bb.%d.%d" % (outf_base, ai, bi)
                yield inf, [outf + '.data', 
                            outf + '.latent', outf + '.meta'], alpha, beta
        
@follows(data_retina_adj_bin)
@files(create_latents_bb_params)
def create_latents_bb(infile, 
                      (data_filename, latent_filename, meta_filename), 
                      alpha, beta):

    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']['link']
    
    model_name= "BetaBernoulli" 

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'alpha' : alpha,
           'beta' : beta}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))



def create_latents_edp_params():
    inf = td('retina.count.data.pickle')
    outf_base = inf[:-len('.data.pickle')]
    outf = "%s.edp" % (outf_base)
    yield inf, [outf + '.data', 
                outf + '.latent', outf + '.meta']
        
@follows(data_retina_adj_count)
@files(create_latents_edp_params)
def create_latents_edp(infile, 
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "ExponentialDistancePoisson"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'rate_scale_hp' : 10.0, 
           'mu_hp': 10.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))



@transform(data_retina_adj_bin, suffix(".data.pickle"), 
           [".ld.truth.data", ".ld.truth.latent", ".ld.truth.meta"])
def create_latents_ld_truth((infile, ),
                      (data_filename, latent_filename, meta_filename)):
    print "Creating latent init to truth", data_filename
    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "LogisticDistance" 

    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types']


    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'mu_hp' : 10.0,
           'lambda_hp' : 10.0,
           'p_min' : 0.02, 
           'p_max' : 0.98}

    irm_latent['relations']['R1']['hps'] = HPS

    irm_latent['domains']['d1']['assignment'] = cell_types




    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    irm.irmio.set_model_latent(irm_model, irm_latent, rng)

    irm.irmio.estimate_suffstats(irm_model, rng, ITERS=20)
    
    pickle.dump(irm.irmio.get_latent(irm_model), 
                open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))


@transform(create_latents_ld_truth, suffix(".truth.data"), 
           [".truth.latent.pdf", ".truth.params.pdf"])
def plot_latents_ld_truth((data_filename, latent_filename, meta_filename), 
                          (output_file, plot_params_filename)):

    data = pickle.load(open(data_filename, 'r'))
    latent = pickle.load(open(latent_filename, 'r'))
    dist_matrix = data['relations']['R1']['data']

    util.plot_latent(latent, dist_matrix, output_file, PLOT_MAX_DIST=120., 
                     MAX_CLASSES=20)
    m = data['relations']['R1']['model']
    ss = latent['relations']['R1']['ss']
    f = pylab.figure()
    ax = f.add_subplot(3, 1, 1)
    ax_xhist = f.add_subplot(3, 1, 2)
    ax_yhist = f.add_subplot(3, 1, 3)

    if m == "LogisticDistance":
        mus_lambs = np.array([(x['mu'], x['lambda']) for x in ss.values()])
    
        ax.scatter(mus_lambs[:, 0], mus_lambs[:, 1], edgecolor='none', 
                   s=2, alpha=0.5)
        ax.set_xlabel('mu')
        ax.set_ylabel('labda')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax_xhist.hist(mus_lambs[:, 0], bins=40)
        ax_xhist.axvline(latent['relations']['R1']['hps']['mu_hp'])
        ax_yhist.hist(mus_lambs[:, 1], bins=40)
        ax_yhist.axvline(latent['relations']['R1']['hps']['lambda_hp'])

    f.suptitle("chain %d for %s" % (0, plot_params_filename))
    
    f.savefig(plot_params_filename)

def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)
            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            
@follows(create_latents_ld_truth)
@follows(create_latents_ld)
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
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
            exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
            yield [data_filename, inits], exp_name, kernel_config_name

@follows(create_inits)
@files(experiment_generator)
def run_exp((data_filename, inits), wait_file, kernel_config_name):
    # put the filenames in the data
    irm.experiments.to_bucket(data_filename, BUCKET_BASE)
    [irm.experiments.to_bucket(init_f, BUCKET_BASE) for init_f in inits]

    kc = KERNEL_CONFIGS[kernel_config_name]
    CHAINS_TO_RUN = len(inits)
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    fixed_k = kc.get('fixed_k', False)
    
    jids = cloud.map(irm.experiments.inference_run, inits, 
                     [data_filename]*CHAINS_TO_RUN, 
                     [kernel_config]*CHAINS_TO_RUN,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     [BUCKET_BASE]*CHAINS_TO_RUN, 
                     [None] * CHAINS_TO_RUN, 
                     [fixed_k] * CHAINS_TO_RUN, 
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

@transform(get_results, suffix(".samples"), [".scoresz.pdf", ".truth.pdf"])
def plot_scores_z(exp_results, (plot_latent_filename, plot_truth_filename)):
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
    cell_id_permutation = d['cell_id_permutation']
    orig_data = pickle.load(open(d['infile']))
    print "len(dist_matrix):", conn.shape
    cell_types = d['types'][:len(conn)]
    # nodes_with_class = meta['nodes']
    # conn_and_dist = meta['conn_and_dist']

    # true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_z = pylab.subplot2grid((2,2), (0, 0))
    ax_score = pylab.subplot2grid((2,2), (0, 1))
    ax_purity =pylab.subplot2grid((2,2), (1, 0), colspan=2)
    ###### zmatrix
    av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
    z = irm.util.compute_zmatrix(av)    

    z_ord = irm.plot.plot_zmatrix(ax_z, z)
    
    ### Plot scores
    for di, d in enumerate(chains):
        subsamp = 2
        s = np.array(d['scores'])[::subsamp]
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

    ax_score.tick_params(axis='both', which='major', labelsize=6)
    ax_score.tick_params(axis='both', which='minor', labelsize=6)
    ax_score.set_xlabel('time (s)')
    ax_score.grid(1)

    purity = irm.experiments.cluster_z_matrix(z > 0.75 * len(chains))
    
    av_idx = np.argsort(purity).flatten()
    ax_purity.scatter(np.arange(len(z)), cell_types[av_idx], s=2)
    newclust = np.argwhere(np.diff(purity[av_idx])).flatten()
    for v in newclust:
        ax_purity.axvline(v)
    ax_purity.set_ylabel('true cell id')
    ax_purity.set_xlim(0, len(z_ord))

    f.tight_layout()

    f.savefig(plot_latent_filename)
    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    synapses = pickle.load(open('synapses.pickle', 'r'))['synapsedf']
    # only take the first 950
    synapses = synapses[(synapses['from_id'] < len(cell_id_permutation) )  & (synapses['to_id']<len(cell_id_permutation))]

    reorder_synapses = util.reorder_synapse_ids(synapses, cell_id_permutation)

    pos_vec = soma_positions['pos_vec'][cell_id_permutation]
    util.plot_cluster_properties(purity, cell_types, pos_vec, reorder_synapses, 
                                 plot_truth_filename)

@transform(get_results, suffix(".samples"), 
           [(".%d.clusters.pdf" % d, ".%d.latent.pdf" % d )  for d in range(1)])
def plot_best_cluster_latent(exp_results, 
                     out_filenames):

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
    conn = d['dist_matrix']['link']
    cell_id_permutation = d['cell_id_permutation']

    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types'][:len(conn)]

    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']
    type_color_map = {'gc' : 'r', 
                      'ac' : 'b', 
                      'bc' : 'g', 
                      'other' : 'k'}

    TYPE_N = np.max(cell_types) + 1

    type_colors = []
    for i in range(TYPE_N):
        if (i < 70):
            d = type_metadata_df.loc[i+1]['desig']
        else:
            d = "  "
        type_colors.append(type_color_map.get(d[:2], 'k'))

    print type_colors 
    

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    synapses = pickle.load(open('synapses.pickle', 'r'))['synapsedf']
    # only take the first 950
    synapses = synapses[(synapses['from_id'] < len(cell_id_permutation) )  & (synapses['to_id']<len(cell_id_permutation))]

    reorder_synapses = util.reorder_synapse_ids(synapses, cell_id_permutation)

    pos_vec = soma_positions['pos_vec'][cell_id_permutation]
    model = data['relations']['R1']['model']

    for chain_pos, (cluster_fname, latent_fname) in enumerate(out_filenames):
        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        cell_assignment = sample_latent['domains']['d1']['assignment']

        a = irm.util.canonicalize_assignment(cell_assignment)

        util.plot_cluster_properties(a, cell_types, 
                                     pos_vec, reorder_synapses, 
                                     cluster_fname, class_colors=type_colors)

        print dist_matrix.dtype, model
        if "istance" not in model:
            dist_matrix = dist_matrix['link']
        util.plot_latent(sample_latent, dist_matrix, latent_fname, 
                         model = model, 
                         PLOT_MAX_DIST=150.0, MAX_CLASSES=20)


        
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

@transform(get_results, suffix(".samples"), [".params.pdf"])
def plot_params(exp_results, (plot_params_filename,)):
    """ 
    plot parmaeters
    """
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    best_chain_i = chains_sorted_order[0]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    
    m = data['relations']['R1']['model']
    ss = sample_latent['relations']['R1']['ss']
    f = pylab.figure()
    ax = f.add_subplot(3, 1, 1)
    ax_xhist = f.add_subplot(3, 1, 2)
    ax_yhist = f.add_subplot(3, 1, 3)
    if m == "LogisticDistance":
        mus_lambs = np.array([(x['mu'], x['lambda']) for x in ss.values()])
        ax.scatter(mus_lambs[:, 0], mus_lambs[:, 1], edgecolor='none', 
                   s=2, alpha=0.5)
        ax.set_xlabel('mu')
        ax.set_ylabel('labda')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)

        ax_xhist.hist(mus_lambs[:, 0], bins=20)
        ax_xhist.axvline(sample_latent['relations']['R1']['hps']['mu_hp'])
        ax_yhist.hist(mus_lambs[:, 1], bins=40)
        ax_yhist.axvline(sample_latent['relations']['R1']['hps']['lambda_hp'])

    f.suptitle("chain %d for %s" % (0, plot_params_filename))
    
    f.savefig(plot_params_filename)

CIRCOS_DIST_THRESHOLDS = [10, 20, 40, 60, 80]

@transform(get_results, suffix(".samples"), 
           [(".circos.%02d.svg" % d, 
             ".circos.%02d.small.svg" % d)  for d in range(len(CIRCOS_DIST_THRESHOLDS))])
def plot_circos_latent(exp_results, 
                       out_filenames):

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
    conn = d['dist_matrix']['link']
    cell_id_permutation = d['cell_id_permutation']

    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types'][:len(conn)]
    
    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    pos_vec = soma_positions['pos_vec'][cell_id_permutation]
    print "Pos_vec=", pos_vec
    model_name = data['relations']['R1']['model']

    # this is potentially fun: get the ranges for each type
    TYPE_N = np.max(cell_types) + 1

    df2 = pandas.DataFrame(index=np.arange(1, TYPE_N))
    df2['des'] = type_metadata_df['coarse']
    df2 = df2.fillna('other')
    df2['id'] = df2.index.values.astype(int)
    gc_mean_i = df2.groupby('des').mean().astype(int)
    gc_min_i = df2.groupby('des').min().astype(int)
    gc_max_i = df2.groupby('des').max().astype(int)



    # create the color map
    # P = np.array([0, gc_max_i.ix['gc'], 
    #               gc_max_i.ix['ac'], 
    #               gc_max_i.ix['bc'], 
    #               TYPE_N], dtype=float)/TYPE_N

    # cdict = {'green': ((P[0], 0.0, 0.0),
    #                 (P[1], 1.0, 1.0),
    #                 (P[2], 1.0, 1.0), 
    #                 (P[3], 1.0, 1.0), 
    #                 (P[4], 1.0, 1.0)),
    #        'blue': ((P[0], 0.0, 0.0),
    #                 (P[1], 0.0, 0.0),
    #                 (P[2], 1.0, 1.0), 
    #                 (P[3], 1.0, 1.0), 
    #                 (P[4], 1.0, 1.0)),
    #         'red': ((P[0], 0.0, 0.0),
    #                 (P[1], 0.0, 0.0),
    #                 (P[2], 0.0, 0.0), 
    #                 (P[3], 1.0, 1.0), 
    #                 (P[4], 1.0, 1.0))}
    # my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    TGT_CMAP = pylab.cm.gist_heat
    coarse_colors = {'other' : [210, 210, 210]}
    for n_i, n in enumerate(['gc', 'nac', 'mwac', 'bc']):
        coarse_colors[n] = colorbrewer.Set1[4][n_i]
    print "THE COARSE COLORS ARE", coarse_colors
    
    for fi, (circos_filename_main, circos_filename_small) in enumerate(out_filenames):
        CLASS_N = len(np.unique(cell_assignment))
        

        class_ids = sorted(np.unique(cell_assignment))

        custom_color_map = {}
        for c_i, c_v in enumerate(class_ids):
            c = np.array(pylab.cm.Set1(float(c_i) / CLASS_N)[:3])*255
            custom_color_map['ccolor%d' % c_v]  = c.astype(int)

        colors = np.linspace(0, 360, CLASS_N)
        color_str = ['ccolor%d' % int(d) for d in class_ids]

        # # add extra colors
        # true_color_list = []
        # for i in range(77):
        #     c = np.array(TGT_CMAP(float(i)/77)[:3])*255
        #     cn = 'true_color_%d' % i
        #     custom_color_map[cn] = c.astype(int)
        #     true_color_list.append(cn)
        
        for n, v in coarse_colors.iteritems():
            custom_color_map['true_coarse_%s' % n] = v

        circos_p = irm.plots.circos.CircosPlot(cell_assignment, 
                                               ideogram_radius="0.7r",
                                               ideogram_thickness="50p", 
                                               karyotype_colors = color_str, 
                                               custom_color_map = custom_color_map)

        if model_name == "LogisticDistance":
            v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
                                               sample_latent['relations']['R1']['ss'], 
                                               sample_latent['relations']['R1']['hps'], 
                                               model_name)
            thold = 0.50 
            ribbons = []
            links = []
            for (src, dest), p in v.iteritems():
                if p > thold:
                    ribbons.append((src, dest, int(40*p)))
            circos_p.set_class_ribbons(ribbons)
            pos_min = 40
            pos_max = 120
            pos_r_min = 1.00
            pos_r_max = pos_r_min + 0.25
            ten_um_frac = 10.0/(pos_max - pos_min)

            circos_p.add_plot('scatter', {'r0' : '%fr' % pos_r_min, 
                                          'r1' : '%fr' % pos_r_max, 
                                          'min' : pos_min, 
                                          'max' : pos_max, 
                                          'glyph' : 'circle', 
                                          'glyph_size' : 10, 
                                          'color' : 'black',
                                          'stroke_thickness' : 0
                                          }, 
                              pos_vec[:, 0], 
                              {'backgrounds' : [('background', {'color': 'vvlgrey', 
                                                                'y0' : pos_min, 
                                                                'y1' : pos_max})],  
                               'axes': [('axis', {'color' : 'vgrey', 
                                                  'thickness' : 1, 
                                                  'spacing' : '%fr' % ten_um_frac})]})
            
            # circos_p.add_plot('heatmap', {'r0' : '1.34r', 
            #                                 'r1' : '1.37r', 
            #                                 'min' : 0, 
            #                                 'max' : 72, 
            #                               'stroke_thickness' : 0, 
            #                               'color' : ",".join(true_color_list) }, 
            #                   cell_types)


            # f_color_legend = pylab.figure()
            # ax_color_legend = f_color_legend.add_subplot(1, 1, 1)

            # x = np.zeros((TYPE_N, 20))
            # for i in range(10):
            #     x[:, i] = np.arange(TYPE_N)
            # for n in ['gc', 'nac', 'mwac', 'bc', 'other']:
            #     print gc_min_i
            #     x[gc_min_i.ix[n]:gc_max_i.ix[n]+1, 10:] = gc_mean_i.ix[n]
            #     ax_color_legend.plot([10, 20], [gc_max_i.ix[n], gc_max_i.ix[n]])
            # ax_color_legend.imshow(x, cmap=TGT_CMAP, interpolation='nearest')
            # ax_color_legend.axvline(10, c='k')
            # ax_color_legend.set_xticks([])
            # f_color_legend.savefig(color_legend_filename)
            print "TYPE_N=", TYPE_N
            type_color_map = {'gc' : 0, 
                              'nac' : 1, 
                              'mwac' : 2, 
                              'bc' : 3, 
                              'other' : 4}

            # pick colors
            colors = ['true_coarse_%s' % s for s in ['gc', 'nac', 'mwac', 'bc', 'other']]
                      
            circos_p.add_plot('heatmap', {'r0' : '1.28r', 
                                          'r1' : '1.34r', 
                                          'min' : 0, 
                                          'max' : 4, 
                                          'stroke_thickness' :0,
                                          'color': ",".join(colors)}, 
                              [type_color_map[df2.ix[i]['des']] for i in cell_types])
            
            # circos_p.add_plot('scatter', {'r0' : '1.01r', 
            #                               'r1' : '1.10r', 
            #                               'min' : 0, 
            #                               'max' : 3, 
            #                               'gliph' : 'square', 
            #                               'color' : 'black', 
            #                               'stroke_thickness' : 0}, 
            #                   [type_lut[i] for i in cell_types])

                            
                                            
        irm.plots.circos.write(circos_p, circos_filename_main)
        
        circos_p = irm.plots.circos.CircosPlot(cell_assignment, ideogram_radius="0.5r", 
                                               ideogram_thickness="80p", 
                                               karyotype_colors = color_str, 
                                               custom_color_map = custom_color_map)
        
        if model_name == "LogisticDistance":
            v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
                                               sample_latent['relations']['R1']['ss'], 
                                               sample_latent['relations']['R1']['hps'], 
                                               model_name)
            thold = 0.50 
            ribbons = []
            links = []
            for (src, dest), p in v.iteritems():
                if p > thold:
                    ribbons.append((src, dest, int(40*p)))
            circos_p.set_class_ribbons(ribbons)
                                            
        irm.plots.circos.write(circos_p, circos_filename_small)

@transform(get_results, suffix(".samples"), 
           ".somapos.pdf")
def plot_clustered_somapos(exp_results, 
                           out_filename):

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
    cell_id_permutation = d['cell_id_permutation']

    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types'][:len(conn)]
    
    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    pos_vec = soma_positions['pos_vec'][cell_id_permutation]

    f = pylab.figure(figsize=(12, 8))
    ax = f.add_subplot(1, 1, 1)

    CLASS_N = len(np.unique(cell_assignment))
    colors = np.linspace(0, 1.0, CLASS_N)

    ca = irm.util.canonicalize_assignment(cell_assignment)
    # build up the color rgb
    cell_colors = np.zeros((len(ca), 3))
    for ci, c in enumerate(ca):
        cell_colors[ci] = pylab.cm.Set1(float(c) / CLASS_N)[:3]
    ax.scatter(pos_vec[:, 1], pos_vec[:, 2], edgecolor='none', 
               c = cell_colors, s=60)
    ax.set_ylim(0, 85)
    ax.set_xlim(5, 115)
    ax.set_aspect(1.0)
    ax.plot([10, 20], [3, 3], linewidth=5, c='k')
    ax.set_xticks([])
    ax.set_yticks([])

    f.savefig(out_filename)

@transform(get_results, suffix(".samples"), 
           ".truth_latent.pdf" )
def plot_truth_latent(exp_results, 
                      out_filename):

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
    conn = d['dist_matrix']['link']
    cell_id_permutation = d['cell_id_permutation']
    
    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types'][:len(conn)]
    
    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    pos_vec = soma_positions['pos_vec'][cell_id_permutation]
    print "Pos_vec=", pos_vec
    model_name = data['relations']['R1']['model']

    # this is potentially fun: get the ranges for each type
    TYPE_N = np.max(cell_types) + 1

    df2 = pandas.DataFrame(index=np.arange(1, TYPE_N))
    df2['des'] = type_metadata_df['coarse']
    df2 = df2.fillna('other')
    df2['id'] = df2.index.values.astype(int)
    gc_mean_i = df2.groupby('des').mean().astype(int)
    gc_min_i = df2.groupby('des').min().astype(int)
    gc_max_i = df2.groupby('des').max().astype(int)

    TGT_CMAP = pylab.cm.gist_heat
    coarse_colors = {'other' : [210, 210, 210]}
    for n_i, n in enumerate(['gc', 'nac', 'mwac', 'bc']):
        coarse_colors[n] = colorbrewer.Set1[4][n_i]
    print "THE COARSE COLORS ARE", coarse_colors

    type_color_map = {'gc' : 0, 
                      'nac' : 1, 
                      'mwac' : 2, 
                      'bc' : 3, 
                      'other' : 4}

    df = pandas.DataFrame({'cell_id' : cell_id_permutation, 
                           'cell_type' : cell_types, 
                           'cluster' : cell_assignment})
    df = df.join(df2, on='cell_type')
    print df.head()

    CLASS_N = len(np.unique(cell_assignment))
    f = pylab.figure(figsize=(8, 11))
    fid = open(out_filename + '.html', 'w')

    # compute the axes positions
    COL_NUMBER = 4
    COL_SPACE = 1.0 / COL_NUMBER
    COL_WIDTH = COL_SPACE - 0.03
    COL_PRE = 0.02
    ROW_CONTRACT = 0.05
    ROW_PRE = 0.02
    ROW_SPACE = 0.01

    VERT_SCALE = 1.0
    s = df['cluster'].value_counts()

    a = irm.util.multi_napsack(COL_NUMBER, np.array(s))
    CLUST_MAX_PER_COL = len(a[0])
    VERT_SCALE = 1.0 -  (CLUST_MAX_PER_COL+4) *ROW_SPACE

    MAX_LEN = np.sum([np.array(s)[ai] for ai in a[0]])

    for col_i, col in enumerate(a):
        pos = 0
        for row_pos in col:
            cluster_id = s.index.values[row_pos]
            sub_df = df[df['cluster'] == cluster_id]
            sub_df = sub_df.sort('cell_type')
            height = len(sub_df) / float(MAX_LEN) * VERT_SCALE

            ax = f.add_axes([COL_PRE + col_i * COL_SPACE, 
                             1.0 - pos - height - ROW_PRE, 
                             COL_WIDTH, height])

            CN = len(sub_df)
            
            for i in range(CN):
                ax.axhline(i, c='k', alpha=0.05)

            colors = [np.array(coarse_colors[ct])/255.0 for ct in sub_df['des']]
            ax.scatter(sub_df['cell_type'], np.arange(CN), 
                       c= colors, s=15,
                       edgecolor='none')
            # optionally plot text
            for i in range(CN):
                t = sub_df.irow(i)['cell_type']
                xpos = 1
                hl = 'left'
                if t < 30:
                    xpos = TYPE_N-2
                    hl = 'right'
                ax.text(xpos, i, "%d" % sub_df.index.values[i], 
                        verticalalignment='center', fontsize=3.5,
                        horizontalalignment=hl)
            
            ax.set_yticks([])
            ax.grid(1)
            ax.set_xlim(0, TYPE_N)
            ax.set_ylim(-0.5, CN+0.5)
            ax.set_xticks([10, 20, 30, 40, 50, 60, 70])


            for tick in ax.xaxis.iter_ticks():
                if pos == 0 :
                    tick[0].label2On = True                    
                tick[0].label1On = False
                tick[0].label2.set_rotation('vertical')
                tick[0].label2.set_fontsize(6) 
            pos += height + ROW_SPACE

    # fid.write(group.to_html())
    fid.close()

    f.savefig(out_filename)


@transform(get_results, suffix(".samples"), 
           ".cluster_metrics.pickle" )
def compute_cluster_metrics(exp_results, 
                      out_filename):

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
    conn = d['dist_matrix']['link']
    cell_id_permutation = d['cell_id_permutation']
    
    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = d['types'][:len(conn)]
    
    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    # this is potentially fun: get the ranges for each type
    TYPE_N = np.max(cell_types) + 1

    df2 = pandas.DataFrame(index=np.arange(1, TYPE_N))
    df2['des'] = type_metadata_df['coarse']
    df2 = df2.fillna('other')
    df2['id'] = df2.index.values.astype(int)
    gc_mean_i = df2.groupby('des').mean().astype(int)
    gc_min_i = df2.groupby('des').min().astype(int)
    gc_max_i = df2.groupby('des').max().astype(int)

    soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    pos_vec = soma_positions['pos_vec'][cell_id_permutation]


    df = pandas.DataFrame({'cell_id' : cell_id_permutation, 
                           'cell_type' : cell_types, 
                           'cluster' : cell_assignment,
                           'x' : pos_vec[:, 0], 
                           'y' : pos_vec[:, 1], 
                           'z' : pos_vec[:, 2]})
    df = df.join(df2, on='cell_type')


    coarse_map = {'gc' : 0, 
                      'nac' : 1, 
                      'mwac' : 2, 
                      'bc' : 3, 
                      'other' : 4}

    canon_true_fine = irm.util.canonicalize_assignment(df['cell_type'])
    canon_true_coarse = [coarse_map[x['des']] for x_i, x in df.iterrows()]
    ca = irm.util.canonicalize_assignment(df['cluster'])


    ari = rand.compute_adj_rand_index(canon_true_fine, ca)
    ari_coarse = rand.compute_adj_rand_index(canon_true_coarse, ca)

                                             
    jaccard = rand.compute_jaccard(canon_true_fine, ca)
    jaccard_coarse = rand.compute_jaccard(canon_true_coarse, ca)

    ss = rand.compute_similarity_stats(canon_true_fine, ca)
    
    # other statistics 
    
    # cluster count
    
    # average variance x
    vars = df.groupby('cluster').var()
    # average variance y
    # average variance z
    
    pickle.dump({'ari' : ari, 
                 'ari_coarse' : ari_coarse, 
                 'jaccard' : jaccard, 
                 'jaccard_coarse' : jaccard_coarse,
                 'n11' : ss['n11'], 
                 'vars' : vars, 
                 'df' : df
                 }, open(out_filename, 'w'))

@merge(compute_cluster_metrics, "cluster_metrics.pickle")
def merge_cluster_metrics(infiles, outfile):
    res = []
    v_df = []
    for infile in infiles:
        d = pickle.load(open(infile, 'r'))
        df = d['df']
        res.append({'filename' : infile, 
                    'ari' : d['ari'], 
                    'ari_coarse' : d['ari_coarse'], 
                    'jaccard_coarse' : d['jaccard_coarse'], 
                    'jaccard' : d['jaccard'], 
                    'n11' : d['n11'], 
                })

        vars = d['vars']
        vars['filename'] = infile
        v_df.append(vars)


    # add in the two others
    fine_vars = df.copy().groupby('cell_type').var()
    fine_vars['filename'] = "truth.fine"

    coarse_vars = df.copy().groupby('des').var()
    coarse_vars['filename'] = "truth.coarse"
    print coarse_vars
    v_df.append(fine_vars)
    v_df.append(coarse_vars)

    clust_df = pandas.DataFrame(res)
    var_df = pandas.concat(v_df)

    pickle.dump({'clust_df' : clust_df, 
                 'var_df' : var_df},
                open(outfile, 'w'))

@files(merge_cluster_metrics, ("spatial_var.pdf", "spatial_var.txt"))
def plot_cluster_vars(infile, (outfile_plot, outfile_rpt)):
    d = pickle.load(open(infile, 'r'))

    var_df = d['var_df']
    var_df = var_df[np.isfinite(var_df['x'])]
    var_df = var_df[np.isfinite(var_df['y'])]
    var_df = var_df[np.isfinite(var_df['z'])]
    tgts = [('Relational Model',
             "1.2.bb.0.0.data-fixed_20_100-anneal_slow_400", 'r', None), 
            ('Spatial-Relational Model', 
             "1.2.ld.0.0.data-fixed_20_100-anneal_slow_400", 'b', None), 
            ('Truth (fine)', 'truth.fine' ,'k', {'linewidth' : 2, 
                                                 'linestyle' : '--'}), 
            ('Truth (coarse)', 'truth.coarse', 'k', {'linewidth' : 4}),
        ]

    f = pylab.figure(figsize=(8,6))
    ax = f.add_subplot(1, 1, 1)
    normed = True
    report_fid = open(outfile_rpt, 'w')
    for t_i, (tgt_name, tgt_fname, c, args) in enumerate(tgts):
        var_df_sub = var_df[var_df['filename'].str.contains(tgt_fname)]

        s = np.sqrt(var_df_sub['y'] + var_df_sub['z'])
        mean = np.mean(s)
        std = np.std(s)
        
        bins = np.linspace(0, 60, 40)

        if 'Truth' not in tgt_name:
            ax.hist(s, bins=bins, 
                    normed=normed, color=c, label=tgt_name)
        else:
            hist, edge = np.histogram(s, bins=bins, normed=normed)
            centers = bins[:-1] + (bins[1] - bins[0])/2.
            
            ax.plot(centers, hist, c=c, label=tgt_name, 
                    **args)
        report_fid.write("%s: mean = %f std=%f \n" % (tgt_name, mean, std))

    ax.set_xlim(0, 60)
    ax.set_xlabel("std. dev. of type (um)")
    ax.set_ylabel("fraction")
    ax.legend(loc="upper left")
    ax.set_title("spatial distribution of type")
    # f = pylab.figure(figsize=(6, 8))
    # for i, v in enumerate(['x', 'y', 'z']):
    #     ax = f.add_subplot(3, 1, i + 1)
    #     vars = []
    #     for t_i, (tgt_name, tgt_fname) in enumerate(tgts):
    #         var_df_sub = var_df[var_df['filename'].str.contains(tgt_fname)]
    #         vars.append(np.sqrt(var_df_sub[v]))

    #     ax.boxplot(vars)
    #     ax.set_xticklabels([x[0] for x in tgts])
    #     ax.set_ylabel("standard dev")
    #     ax.set_title(v)


    f.tight_layout()
    f.savefig(outfile_plot)
    
pipeline_run([data_retina_adj_bin, 
              data_retina_adj_count, 
              create_inits, 
              plot_scores_z, 
              plot_best_cluster_latent, 
              #plot_hypers, 
              plot_latents_ld_truth, 
              plot_params, 
              create_latents_ld_truth, 
              plot_circos_latent, 
              plot_clustered_somapos,
              plot_truth_latent, 
              compute_cluster_metrics, 
              merge_cluster_metrics,
              plot_cluster_vars
          ], multiprocess=2)
                        
