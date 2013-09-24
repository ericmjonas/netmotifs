from ruffus import *
import cPickle as pickle
import numpy as np
import pandas
import copy
import os, glob
import time
from matplotlib import pylab
from jinja2 import Template
import irm
import irm.data
import matplotlib.gridspec as gridspec
from matplotlib import colors
import copy


def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


import cloud

BUCKET_BASE="srm/experiments/celegans"

WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)


EXPERIMENTS = [
               ('celegans.2r.bb.00', 'fixed_100_200', 'anneal_slow_400'),  
    ('celegans.2r.gp.00', 'fixed_100_200', 'anneal_slow_400'),  
    ('celegans.2r.ld.00', 'fixed_100_200', 'anneal_slow_400'),  
    ('celegans.2r.sdb.00', 'fixed_100_200', 'anneal_slow_400'),  
    ('celegans.2r.ndfw.00', 'fixed_100_200', 'anneal_slow_400'),  

           ]

INIT_CONFIGS = {'fixed_10_100' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
                'crp_100_20' : {'N' : 100, 
                               'config' : {'type' : 'crp', 
                                           'alpha' : 20.0}}
}

                               
                
                

BB_HPS = [(0.1, 0.1)]# , (1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
GP_HPS = [(1.0, 2.0)]# , (2.0, 2.0), (3.0, 2.0)]
LD_HPS = [(0.1, 0.1, 0.9, 0.01)] # , (0.1, 0.1, 0.5, 0.001),
NDFW_HPS = [(0.1, 0.001, 0.1, 1.0, 1.0)]
SDB_HPS = [(1.0, 1.0, 0.1, 0.001, 0.5, 4.0)]
default_nonconj = irm.runner.default_kernel_nonconj_config()
default_conj = irm.runner.default_kernel_config()
default_anneal = irm.runner.default_kernel_anneal()
slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300
slow_anneal[0][1]['subkernels'][-2][1]['grid'] = np.linspace(1.0, 50.0, 50.0)

def generate_ld_hypers():
    mu_min = 0.01
    mu_max = 0.9
    space_vals =  irm.util.logspace(mu_min, mu_max, 50)
    p_mins = np.array([0.001])
    p_maxs = np.array([0.90]) # , 0.80, 0.50, 0.20])
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for p_max in p_maxs:
                res.append({'lambda_hp' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 'p_max' : p_max})
    return res

slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistance'] = generate_ld_hypers()


def generate_sdb_hypers():

    p_alphas = irm.util.logspace(0.001, 1.0, 20)
    p_betas = irm.util.logspace(0.5, 10.0, 20)
    mu_hps = irm.util.logspace(0.01, 0.4, 10)
    hps = []
    for a in p_alphas:
        for b in p_betas:
            for mu_hp in mu_hps:
                hps.append({'p_alpha' : a, 'p_beta': b, 
                            'mu_hp' : mu_hp, 
                            'p_min' : 0.001, 
                            'param_weight' : 0.5, 
                            'param_max_distance' : 4.0})
                
    return hps

slow_anneal[0][1]['subkernels'][-1][1]['grids']['SquareDistanceBump'] = generate_sdb_hypers()


def generate_ndfw_hypers():
    p_mins = np.array([0.001, 0.01, 0.02])
    mus = np.array([0.1, 1.0])
    widths = np.array([0.1, 0.5, 1.0])
    alphabetas = [0.1, 1.0, 2.0]
    hps = []
    for p_min in p_mins:
        for mu in mus:
            for width in widths:
                for a in alphabetas:
                    for b in alphabetas:
                        hps.append({'mu_hp' : mu, 
                                    'p_min' : p_min, 
                                    'width' : width, 
                                    'p_alpha' : a,
                                    'p_beta' : b})
    return hps

slow_anneal[0][1]['subkernels'][-1][1]['grids']['NormalDistanceFixedWidth'] = generate_ndfw_hypers()

slow_anneal[0][1]['subkernels'][-1][1]['grids']['BetaBernoulli'] =  [{'alpha' : a, 'beta' : b} for a, b in irm.util.cart_prod([irm.util.logspace(0.001, 1.0, 20), irm.util.logspace(0.1, 10, 20)])]

KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},

                  }




@split('data.processed.pickle', ['celegans.both.data.pickle', 
                                 'celegans.electrical.data.pickle', 
                                 'celegans.chemical.data.pickle'])
def data_celegans_adj(infile, (both_file, electrical_file, chemical_file)):
    data = pickle.load(open(infile, 'r'))
    conn_matrix = data['conn_matrix']
    neurons = data['neurons']
    canonical_neuron_ordering = data['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)
    dist_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])
    # compute distance
    for n1_i, n1 in enumerate(canonical_neuron_ordering):
        for n2_i, n2 in enumerate(canonical_neuron_ordering):
            dist_matrix[n1_i, n2_i]['distance'] = np.abs(neurons.loc(n1)['soma_pos'] - neurons.loc(n2)['soma_pos'])
    
    adj_mat_chem = conn_matrix['chemical'] > 0
    adj_mat_elec = conn_matrix['electrical'] > 0
    adj_mat_both = np.logical_or(adj_mat_chem, adj_mat_elec)
    
    dist_matrix['link'] = adj_mat_both
    pickle.dump({'dist_matrix' : dist_matrix, 
                 'infile' : infile}, open(both_file, 'w'))
    
    dist_matrix['link'] = adj_mat_chem
    pickle.dump({'dist_matrix' : dist_matrix, 
                 'infile' : infile}, open(chemical_file, 'w'))
    
    dist_matrix['link'] = adj_mat_elec
    pickle.dump({'dist_matrix' : dist_matrix, 
                 'infile' : infile}, open(electrical_file, 'w'))
    
    
    


def create_latents_2r_param():
    infile = 'data.processed.pickle'
    for bb_hpi in range(len(BB_HPS)):
        base = td('celegans.2r.bb.%02d' % bb_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'BetaBernoulli', bb_hpi
    for gp_hpi in range(len(GP_HPS)):
        base = td('celegans.2r.gp.%02d' % gp_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'GammaPoisson', gp_hpi

    for ld_hpi in range(len(LD_HPS)):
        base = td('celegans.2r.ld.%02d' % ld_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'LogisticDistance', ld_hpi

    for sdb_hpi in range(len(SDB_HPS)):
        base = td('celegans.2r.sdb.%02d' % sdb_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'SquareDistanceBump', sdb_hpi

    for ndfw_hpi in range(len(NDFW_HPS)):
        base = td('celegans.2r.ndfw.%02d' % ndfw_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'NormalDistanceFixedWidth', ndfw_hpi
        
@files(create_latents_2r_param)
def create_latents_2r_paramed(infile, 
                              (data_filename, latent_filename, meta_filename), 
                              model_name, hp_i):


    data = pickle.load(open(infile, 'r'))
    conn_matrix = data['conn_matrix']
    neurons = data['neurons']
    canonical_neuron_ordering = data['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)
    dist_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

    for n1_i, n1 in enumerate(canonical_neuron_ordering):
        for n2_i, n2 in enumerate(canonical_neuron_ordering):
            dist_matrix[n1_i, n2_i] = np.abs(neurons.loc[n1]['soma_pos'] - neurons.loc[n2]['soma_pos'])

    if model_name == "BetaBernoulli":
        chem_conn = conn_matrix['chemical'] > 0 

        elec_conn = conn_matrix['electrical'] > 0 

        irm_latent, irm_data = irm.irmio.default_graph_init(elec_conn, model_name, 
                                                            extra_conn=[chem_conn])

        HPS = {'alpha' : BB_HPS[hp_i][0], 
               'beta' : BB_HPS[hp_i][1]}
 
    elif model_name == "LogisticDistance":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : LD_HPS[hp_i][0], 
               'lambda_hp' : LD_HPS[hp_i][1], 
               'p_max' : LD_HPS[hp_i][2], 
               'p_min' : LD_HPS[hp_i][3]}
    elif model_name == "NormalDistanceFixedWidth":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : NDFW_HPS[hp_i][0], 
               'p_min' : NDFW_HPS[hp_i][1], 
               'width' : NDFW_HPS[hp_i][2], 
               'p_alpha' : NDFW_HPS[hp_i][3], 
               'p_beta' : NDFW_HPS[hp_i][4]}

    elif model_name == "SquareDistanceBump":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'p_alpha' : SDB_HPS[hp_i][0], 
               'p_beta' : SDB_HPS[hp_i][1], 
               'mu_hp' : SDB_HPS[hp_i][2], 
               'p_min' : SDB_HPS[hp_i][3], 
               'param_weight' : SDB_HPS[hp_i][4], 
               'param_max_distance' : SDB_HPS[hp_i][5]}

    elif model_name == "GammaPoisson":
        chem_conn = conn_matrix['chemical'].astype(np.uint32)

        elec_conn = conn_matrix['electrical'].astype(np.uint32)

        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'alpha' : GP_HPS[hp_i][0], 
               'beta' : GP_HPS[hp_i][1]}
               
    
    irm_latent['relations']['R1']['hps'] = HPS
    irm_latent['relations']['R2']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 'r1' : chem_conn, 
                 'r2' : elec_conn},
                open(meta_filename, 'w'))


def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name,  i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            

@follows(create_latents_2r_paramed)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    irm.experiments.create_init(latent_filename, data_filename, out_filenames, 
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
    
    jids = cloud.map(irm.experiments.inference_run, inits, 
                     [data_filename]*CHAINS_TO_RUN, 
                     [kernel_config]*CHAINS_TO_RUN,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     [BUCKET_BASE]*CHAINS_TO_RUN, 
                     _env='connectivitymotif', 
                     _label="%s-%s-%s" % (data_filename, inits[0], 
                                          kernel_config_name), 
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
    if isinstance(meta_infile, list):
        # hack to correct for the fact that multi-relation datasets have multiple infiles. Should fix 
        meta_infile = meta_infile[0] 

    d = pickle.load(open(meta_infile, 'r'))
    if 'infile' not in d: # And this gross hack is due to our parametric exploration 
        # of the hypers above, where we directly generate the .data from the raw source=
        orig_processed_data = d
    else:
        very_original_data = d['infile']
        orig_processed_data = pickle.load(open(very_original_data, 'r'))
    canonical_neuron_ordering = orig_processed_data['canonical_neuron_ordering']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_z = pylab.subplot2grid((2,2), (0, 0))
    ax_score = pylab.subplot2grid((2,2), (0, 1))


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

@transform(get_results, suffix(".samples"), 
           [(".%d.clusters.pdf" % d, )  for d in range(1)])
def plot_best_cluster(exp_results, 
                     out_filenames):
    from matplotlib.backends.backend_pdf import PdfPages

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']
    #print "meta_infile=", meta_infile

    if isinstance(meta_infile, list): # hack to correct for the fact that multi-relation datasets have multiple infiles. Should fix 
        meta_infile = meta_infile[0] 


    d = pickle.load(open(meta_infile, 'r'))

    if 'infile' not in d: # And this gross hack is due to our parametric exploration 
        # of the hypers above, where we directly generate the .data from the raw source=
        orig_processed_data = d
    else:
        very_original_data = d['infile']
        orig_processed_data = pickle.load(open(very_original_data, 'r'))

    canonical_neuron_ordering = orig_processed_data['canonical_neuron_ordering']
    no = np.array(canonical_neuron_ordering)

    neurons = orig_processed_data['neurons']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (cluster_fname, ) in enumerate(out_filenames):


        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        
        # irm.experiments.plot_latent(sample_latent, dist_matrix, 
        #                             latent_fname, #cell_types, 
        #                             #types_fname, 
        #                             model=data['relations']['R1']['model'], 
        #                             PLOT_MAX_DIST=1.2)
        a = irm.util.canonicalize_assignment(sample_latent['domains']['d1']['assignment'])
        fig = plot_clusters_pretty_figure(a, neurons, no)
        fig.savefig(cluster_fname)

# @transform(get_results, suffix(".samples"), [".clusters.html"])
# def cluster_interpretation(exp_results, (output_filename,)):
#     sample_d = pickle.load(open(exp_results))
#     chains = sample_d['chains']
    
#     exp = sample_d['exp']
#     data_filename = exp['data_filename']
#     data = pickle.load(open(data_filename))
#     data_basename, _ = os.path.splitext(data_filename)
#     meta = pickle.load(open(data_basename + ".meta"))

#     meta_infile = meta['infile']
#     if isinstance(meta_infile, list): # hack to correct for the fact that multi-relation datasets have multiple infiles. Should fix 
#         meta_infile = meta_infile[0] 

#     d = pickle.load(open(meta_infile, 'r'))

#     if 'infile' not in d: # And this gross hack is due to our parametric exploration 
#         # of the hypers above, where we directly generate the .data from the raw source=
#         orig_processed_data = d
#     else:
#         very_original_data = d['infile']
#         orig_processed_data = pickle.load(open(very_original_data, 'r'))
#     canonical_neuron_ordering = orig_processed_data['canonical_neuron_ordering']

#     chains = [c for c in chains if type(c['scores']) != int]
#     CHAINN = len(chains)

#     ###### zmatrix
#     av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
#     z = irm.util.compute_zmatrix(av)    

#     purity = irm.experiments.cluster_z_matrix(z, ITERS=10, method='dpmm_bb')
#     av_idx = np.argsort(purity).flatten()
#     no = np.array(canonical_neuron_ordering)

#     template = Template(open("report.template", 'r').read())
#     neuron_data = orig_processed_data['neurons']

#     df = pandas.io.excel.read_excel("../../../data/celegans/manualmetadata.xlsx", 
#                                     'properties', 
#                                     index_col=0)
#     print df.head()

#     clusters = []
#     f = pylab.figure(figsize=(8, 2))
#     ax = f.add_subplot(1, 1, 1)
#     for cluster_i in range(len(np.unique(purity))):
#         cluster = []
#         def nc(x):
#             print type(x)
#             if isinstance(x, unicode):
#                 return x
#             if np.isnan(x):
#                 return ""
#             return x
        
#         for n_i, n in enumerate(no[cluster_i == purity]):
#             cluster.append({'id' : n, 
#                             'neurotransmitters' : nc(df.loc[n]['neurotransmitters']), 
#                             'role' : nc(df.loc[n]['role']), 
#                             'basic' : nc(df.loc[n]['basic']), 
#                             'extended' : nc(df.loc[n]['extended'])})
#         cluster.sort(key=lambda x: x['id'])
#         cluster_size = len(cluster)

#         positions = [neuron_data[n['id']]['soma_pos'] for n in cluster]
#         ax.plot([0.0, 1.0], [0.5, 0.5], linewidth=1, c='k')
#         ax.scatter(positions, np.random.normal(0, 0.01, cluster_size)  + 0.5, 
#                    c='r', s=12, 
#                    edgecolor='none')
#         ax.set_xlim(-0.05, 1.05)
#         ax.set_ylim(0.3, 0.7)
#         ax.set_yticks([])
        
#         fig_fname = "%s.%d.somapos.png" % (output_filename, cluster_i)
#         f.savefig(fig_fname)
#         ax.cla()
        
#         clusters.append({'neurons' : cluster, 
#                          'soma_pos_file' : fig_fname})

#     fid = open(output_filename, 'w')
#     fid.write(template.render(clusters=clusters))
    

    
    # ax_purity.scatter(np.arange(len(z)), cell_types[av_idx], s=2)
    # newclust = np.argwhere(np.diff(purity[av_idx])).flatten()
    # for v in newclust:
    #     ax_purity.axvline(v)
    # ax_purity.set_ylabel('true cell id')
    # ax_purity.set_xlim(0, len(z_ord))

@transform(get_results, suffix(".samples"), [".clusters.pdf"])
def cluster_interpretation_plot(exp_results, (output_filename,)):
    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    thold = 0.9 

    meta_infile = meta['infile']
    if isinstance(meta_infile, list): # hack to correct for the fact that multi-relation datasets have multiple infiles. Should fix 
        meta_infile = meta_infile[0] 

    d = pickle.load(open(meta_infile, 'r'))

    if 'infile' not in d: # And this gross hack is due to our parametric exploration 
        # of the hypers above, where we directly generate the .data from the raw source=
        orig_processed_data = d
    else:
        very_original_data = d['infile']
        orig_processed_data = pickle.load(open(very_original_data, 'r'))
    canonical_neuron_ordering = orig_processed_data['canonical_neuron_ordering']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    ###### zmatrix
    av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
    z = irm.util.compute_zmatrix(av)    

    purity = irm.experiments.cluster_z_matrix(z, ITERS=10, method='dpmm_gp')



    no = np.array(canonical_neuron_ordering)

    neuron_data = orig_processed_data['neurons']

    df = orig_processed_data['neurons']
    fig = plot_clusters_pretty_figure(purity, df, no, thold=thold)

    fig.savefig(output_filename)

def plot_clusters_pretty_figure(purity, metadata_df, no, thold=0.9):
    df = metadata_df

    purity = irm.util.canonicalize_assignment(purity)
    CLASSES = np.sort(np.unique(purity))

    CLASSN = len(CLASSES)

    class_sizes = np.zeros(CLASSN)
    for i in range(CLASSN):
        class_sizes[i] = np.sum(purity == i)
    height_ratios = class_sizes / np.sum(class_sizes)

    w = np.argwhere(np.cumsum(height_ratios) <= thold).flatten()
    if len(w) == 0:
        CLASSN_TO_PLOT = 1
    else:
        CLASSN_TO_PLOT = w[-1] + 1

    clusters = []
    fig = pylab.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(2, CLASSN_TO_PLOT,
                           width_ratios=height_ratios, 
                           height_ratios=[0.6, 0.3])
    NEURON_CLASSES = ['AS', 'DA', 'DB', 'DD', 'VA', 'VB', 'VD', 
                      'IL1', 'IL2', 'OLQ', 'RMD', 'SAA', 'SIB', 'SMD', 
                      'URA', 'URB', 'URY']
    NEURON_CLASSES_ROLES = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 
                          '', '', 'S', '', 'S', '', 'S', 
                          'S', 'S', 'S']
    neuron_classes_size = {}
    # compute sizes 
    for nc in NEURON_CLASSES:
        neuron_classes_size[nc] = 0
        for rn, r in df.iterrows():
            if rn.startswith(nc): neuron_classes_size[nc] +=1
        
    for cluster_i in range(len(np.unique(purity))):
        if cluster_i >= CLASSN_TO_PLOT:
            continue 

        cluster = []
        def nc(x):
            if isinstance(x, unicode):
                return x
            if np.isnan(x):
                return ""
            return x

        motor_neuron_count = 0
        sensory_neuron_count = 0
        roles = []
        cluster_neuron_classes = {nc : 0 for nc in NEURON_CLASSES}
        for n_i, n in enumerate(no[cluster_i == purity]):
            role =  nc(df.loc[n]['role'])
            cluster.append({'id' : n, 
                            'neurotransmitters' : nc(df.loc[n]['neurotransmitters']), 
                            'role' : role, 
                            'basic' : nc(df.loc[n]['basic']), 
                            'extended' : nc(df.loc[n]['extended'])})
            for ncl in NEURON_CLASSES:
                if n.startswith(ncl):
                    cluster_neuron_classes[ncl] += 1

            roles.append(role)
            
        roles = np.array(roles)
        cluster.sort(key=lambda x: x['id'])
        cluster_size = len(cluster)
        role_color_lut = {'' : '#AAAAAA', 'M' : 'b', 'S' : 'g'}

        colors = [role_color_lut[r] for r in roles]
        
        positions = [df.loc[n['id']]['soma_pos'] for n in cluster]
        ax = fig.add_subplot(gs[0, cluster_i])

        ax.plot([0.5, 0.5], [0.0, 1.0], linewidth=1, c='k')
        ax.scatter(np.random.normal(0, 0.01, cluster_size)  + 0.5, 
                   positions ,
                   c=colors, s=12, 
                   edgecolor='none')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        

        m_frac = np.sum(roles == 'M')  / float(len(cluster))
        s_frac = np.sum(roles == 'S') / float(len(cluster))
        other_frac = 1.0 - s_frac -m_frac
        BAR_H = 0.03
        ax.barh(-0.05, m_frac, height=BAR_H, left=0.0, color='b')
        ax.barh(-0.05, m_frac + s_frac, height=BAR_H, left = m_frac, color='g')
        ax.barh(-0.05, 1.0, height=BAR_H, left = m_frac + s_frac, color="#AAAAAA")


        ax_nc = fig.add_subplot(gs[1, cluster_i])
        # generate the bars:
        bars = []
        bar_colors = []
        for nc, nc_type in zip(NEURON_CLASSES[::-1], NEURON_CLASSES_ROLES[::-1]):
            nc_size =  float(neuron_classes_size[nc])
            if nc_size > 0:
                bars.append(cluster_neuron_classes[nc] / float(neuron_classes_size[nc]))
            else:
                bars.append(0)
            bar_colors.append(role_color_lut[nc_type])
        ax_nc.barh(np.arange(len(NEURON_CLASSES)), 
                    bars, color=bar_colors)
        ax_nc.set_xticks([])
        if cluster_i == 0:
            ax_nc.set_yticks(np.arange(len(NEURON_CLASSES)) + 0.4)
            ax_nc.set_yticklabels(NEURON_CLASSES[::-1], fontsize='xx-small')
        else:
            ax_nc.set_yticks([])
        ax_nc.set_ylim(0, len(NEURON_CLASSES))
        ax_nc.set_xlim(0, 1.0)


        clusters.append({'neurons' : cluster})
    return fig
#FIXME other metrics: L/R successful grouping
#FIXME other metrics: *n grouping

@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, )  for d in range(1)])
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

    # d = pickle.load(open(meta_infile, 'r'))
    # dist_matrix = d['dist_matrix']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)
    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (latent_fname, ) in enumerate(out_filenames):
        print "plotting chain", chain_pos

        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']

        model = data['relations']['R1']['model']
        
        f = pylab.figure(figsize= (24, 26))
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,12])
        ax = f.add_subplot(gs[1, 0])

        cmap = colors.ListedColormap(['white', 'red', 'blue', 
                                      'purple'])
        bounds=[0,1, 2, 3, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if model == "BetaBernoulli":
            data_mat = data['relations']['R1']['data'].astype(int)

            data_mat += data['relations']['R2']['data'].astype(int)*2

        if model == "GammaPoisson":
            data_mat = data['relations']['R1']['data']
        elif model == "LogisticDistance" or model == "SquareDistanceBump" or model == "NormalDistanceFixedWidth":
            data_mat = data['relations']['R1']['data']['link'].astype(int)
            data_mat += data['relations']['R2']['data']['link'].astype(int)*2
        a = sample_latent['domains']['d1']['assignment']
        irm.plot.plot_t1t1_latent(ax, data_mat, a, cmap=cmap, norm=norm)
                                  
        
        f.savefig(latent_fname)

pipeline_run([create_inits, get_results, plot_scores_z, 
              plot_best_latent, 
              plot_best_cluster, 
              cluster_interpretation_plot, 
              plot_hypers
          ], multiprocess=3)
                        
