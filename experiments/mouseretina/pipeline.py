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
import util

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


import cloud

BUCKET_BASE="srm/experiments/mouseretina"


EXPERIMENTS = [#('retina.bb', 'fixed_100_200', 'default_10'), 
               #('retina.bb', 'fixed_100_200', 'default_100'), 
               ('retina.0.0.ld.0.0', 'fixed_10_20', 'default_nc_10'),
               # ('retina.1.1.ld.0.0', 'fixed_100_200', 'default_nc_1000'), 
               # ('retina.1.0.ld.0.0', 'fixed_100_200', 'default_nc_1000'), 
               # ('retina.1.1.ld.1.0', 'fixed_100_200', 'default_nc_1000'), 
               # ('retina.1.0.ld.1.0', 'fixed_100_200', 'default_nc_1000'), 
               #('retina.ld', 'fixed_100_200', 'default_nc_100')
               #('retina.bb', 'fixed_100_200', 'default_10000'), 
           ]

THOLDS = [0.01, 0.1, 0.5, 1.0]
    
MULAMBS = [1.0, 5.0, 10.0, 20.0, 50.0]
PMAXS = [0.95, 0.9, 0.7, 0.5]


# for i in range(len(THOLDS)):
#     for z in [0, 1]:
#         for k in range(len(MULAMBS)):
#             for l in range(len(PMAXS)):
#                 EXPERIMENTS.append(('retina.%d.%d.ld.%d.%d' % (i, z, k, l), 
#                                     'fixed_10_200',
#                                     'default_nc_10'))
            
#                 EXPERIMENTS.append(('retina.%d.%d.ld.%d.%d' % (i, z, k, l), 
#                                     'fixed_100_200',
#                                     'default_nc_100'))
            
            
INIT_CONFIGS = {'fixed_10_200' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
                'fixed_10_20' : {'N' : 10, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 20}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}}


default_nonconj = irm.runner.default_kernel_nonconj_config()
default_conj = irm.runner.default_kernel_config()


KERNEL_CONFIGS = {'default_nc_100' : {'ITERS' : 100, 
                                  'kernels' : default_nonconj},
                  'default_nc_10' : {'ITERS' : 10, 
                                  'kernels' : default_nonconj},
                  'default_10' : {'ITERS' : 10, 
                                  'kernels' : default_conj},
                  'default_100' : {'ITERS' : 100, 
                                  'kernels' : default_conj},
                  'default_nc_1000' : {'ITERS' : 1000, 
                                       'kernels' : default_nonconj},
                  }


def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))

def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))

def create_tholds():
    """
    systematicaly vary the threshold for "synapse" and whether or not
    we use the z-axis
    """
    infiles = ['rawdata.pickle', 'xlsxdata.pickle', 
               'soma.positions.pickle']
    for use_z in [0, 1]:
        for tholdi, thold in enumerate(THOLDS):
            outfile = "retina.%d.%d.data.pickle" % (use_z, tholdi)
            yield infiles, [outfile], thold, use_z

@files(create_tholds)
def data_retina_adj((raw_infile, xlsx_infile, positions_infile), 
                    (retina_outfile,), AREA_THOLD, USE_Z):
    data = pickle.load(open(xlsx_infile, 'r'))
    area_mat = data['area_mat']
    positions_data = pickle.load(open(positions_infile, 'r'))
    pos_vec = positions_data['pos_vec']
    NEURON_N = 950 # only the ones for which we also have position data

    area_mat_sub = area_mat[:NEURON_N, :NEURON_N]

    dist_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

    dist_matrix['link'] = area_mat_sub > AREA_THOLD
    for n1 in range(NEURON_N):
        for n2 in range(NEURON_N):
            p1 = pos_vec[n1]
            p2 = pos_vec[n2]
            if not USE_Z:
                p1[2] = 0
                p2[2] = 0
            
            dist_matrix[n1, n2]['distance'] = dist(p1, p2)

    pickle.dump({'dist_matrix' : dist_matrix, 
                 'area_thold' : AREA_THOLD, 
                 'infile' : xlsx_infile}, open(retina_outfile, 'w'))
                
def create_latents_ld_params():
    for a in create_tholds():
        inf = a[1][0]
        for mli, mulamb in enumerate(MULAMBS):
            for pi, p in enumerate(PMAXS):
                outf_base = inf[:-len('.data.pickle')]
                outf = "%s.ld.%d.%d" % (outf_base, mli, pi)
                yield inf, [outf + '.data', 
                            outf + '.latent', outf + '.meta'], mulamb, p
        
    
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


@transform(data_retina_adj, suffix(".data.pickle"), 
           [".bb.data", ".bb.latent", ".bb.meta"])
def create_latents_bb((infile, ),
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


def create_init(latent_filename, data_filename, out_filenames, 
                init= None):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    irm_data = pickle.load(open(data_filename, 'r'))
    irm_latents = []

    rng = irm.RNG()

    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    for c, out_f in enumerate(out_filenames):
        print "generating init", out_f
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

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)

        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=10)


        pickle.dump(irm.irmio.get_latent(irm_model), open(out_f, 'w'))


def get_dataset(data_name):
    return glob.glob("%s.data" %  data_name)

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            

@follows(create_latents_ld)
@follows(create_latents_bb)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    create_init(latent_filename, data_filename, 
                out_filenames, 
                init= init_config['config'])



def inference_run(latent_filename, 
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

            inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
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
    
    jids = cloud.map(inference_run, inits, 
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
    cell_types = orig_data['types'][:len(conn)]

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
        subsamp = 4
        s = np.array(d['scores'])[::subsamp]
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

    ax_score.tick_params(axis='both', which='major', labelsize=6)
    ax_score.tick_params(axis='both', which='minor', labelsize=6)
    ax_score.set_xlabel('time (s)')
    ax_score.grid(1)


    ax_purity.plot(cell_types[z_ord])
    ax_purity.set_ylabel('true cell id')

    f.tight_layout()

    f.savefig(plot_latent_filename)
    f = pylab.figure(figsize=(20, 4))



@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, ".%d.types.pdf" % d)  for d in range(3)])
def plot_best_latent(exp_results, 
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
    print "meta_infile=", meta_infile

    d = pickle.load(open(meta_infile, 'r'))
    conn = d['dist_matrix']['link']
    dist_matrix = d['dist_matrix']
    orig_data = pickle.load(open(d['infile']))
    cell_types = orig_data['types'][:len(conn)]

    type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']
    
    # nodes_with_class = meta['nodes']
    # conn_and_dist = meta['conn_and_dist']

    # true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (latent_fname, types_fname) in enumerate(out_filenames):


        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        
        util.plot_latent(sample_latent, dist_matrix, 
                         latent_fname, cell_types, 
                         types_fname)

        # a = np.array(sample_latent['domains']['d1']['assignment'])

        # f = pylab.figure(figsize= (24, 26))

        # gs = gridspec.GridSpec(2, 1, height_ratios=[1,12])


        # ax = pylab.subplot(gs[1])
        # ax_types = pylab.subplot(gs[0])

        # ai = irm.plot.plot_t1t1_latent(ax, conn, a)


        # # gross_types = np.zeros_like(cell_types)
        # # gross_types[:12] = 0
        # # gross_types[12:57] = 1
        # # gross_types[58:] = 2 

        # # cluster_types = irm.util.compute_purity(a, gross_types)
        # # for k, v in cluster_types.iteritems():
        # #     print k, ":",  v

        # for i in  np.argwhere(np.diff(a[ai]) != 0):
        #     ax_types.axvline(i, c='b', alpha=0.7, linewidth=1.0)

        # ax_types.scatter(np.arange(len(cell_types)), 
        #                  cell_types[ai], edgecolor='none', c='k', 
        #                  s=2)

        # ax_types.set_xlim(0, len(cell_types))
        # ax_types.set_ylim(0, 80)
        # ax_types.set_xticks([])
        # ax_types.set_yticks([])

        # f.tight_layout()
        # pp = PdfPages(latent_fname)
        # f.savefig(pp, format='pdf')


        # f2 =  pylab.figure(figsize= (12, 12))
        # irm.plot.plot_t1t1_params(f2, dist_matrix, a, 
        #                           sample_latent['relations']['R1']['ss'], 
        #                           MAX_DIST=200)
        # f2.savefig(pp, format='pdf')
        # pp.close()


        # f = pylab.figure()
        # ax_types = pylab.subplot(1, 1, 1)
        # irm.plot.plot_purity_ratios(ax_types, a, cell_types)


        # f.savefig(types_fname)
    

pipeline_run([data_retina_adj, create_latents_bb, plot_scores_z, 
              plot_best_latent], multiprocess=2)
                        
