from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os
import time

import irm
import irm.data
from matplotlib import pylab
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Grid

import cloud
import rand

BUCKET_BASE="srm/experiments/testinf/"

#cloud.start_simulator()

def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

"""
for N connection classes 
for M seeds
    generate the data
    run the regular IRM on it
    run our iIRM on it with 10 different rng inits, 10 different seeds

"""

def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))

CHAINS_TO_RUN = 10
SAMPLER_ITERS = 50
SEEDS = np.arange(2)

JITTERS = [0, 0.01, 0.1]

INITIAL_GROUP_NUM = 40

#SKIP = 100
#BURN = 700
def data_generator():

    POSSIBLE_SIDE_N = [10]
    
    conn = {
        'twoclass': {(0, 1) : (2.0, 0.7), 
                         (1, 0) : (3.0, 0.8)},
            'oneclass': {(0, 0) : (2.0, 0.8)}, 
            '3c0' : {(0, 1) : (1.5, 0.5), 
                    (1, 2) : (2.0, 0.3), 
                    (2, 0) : (3.0, 0.8)}, 
            '3c1' : {(0, 1) : (2.0, 0.5), 
                     (1, 2) : (2.0, 0.3), 
                     (0, 2) : (2.0, 0.7),                      
                     (2, 0) : (2.0, 0.8),
                 },
            # much lower probs
            '3c2' : {(0, 1) : (2.0, 0.2), 
                     (1, 2) : (2.0, 0.1), 
                     (0, 2) : (2.0, 0.4),                      
                     (2, 0) : (2.0, 0.2),
                 },
            # change the radius
            '3c3' : {(0, 1) : (1.0, 0.5), 
                     (1, 2) : (1.5, 0.3), 
                     (0, 2) : (2.0, 0.7),                      
                     (2, 0) : (2.5, 0.8),
                 },
            '5c0' : {(0, 1) : (1.0, 0.5), 
                     (1, 2) : (1.5, 0.3), 
                     (0, 2) : (2.0, 0.7),                      
                     (2, 0) : (2.5, 0.8),
                     (2, 3) : (2.5, 0.8),
                     (3, 4) : (2.5, 0.8),
                     (1, 4) : (1.5, 0.4),
                 },
            '10c2' : {(0, 1) : (1.0, 0.5), 
                      (0, 2) : (2.0, 0.7),                      
                      (0, 8) : (2.0, 0.7), 
                      (0, 3) : (1.0, 0.7),                      
                      (1, 4) : (1.5, 0.4),
                      (1, 2) : (1.5, 0.3), 
                      (1, 4) : (1.5, 0.4),
                      (2, 0) : (2.5, 0.6),
                      (2, 3) : (2.1, 0.8),
                      (2, 0) : (2.5, 0.6),
                      (2, 7) : (3.0, 0.7),
                      (3, 1) : (0.9, 0.7), 
                      (3, 4) : (2.5, 0.8),
                      (3, 6) : (2.2, 0.4), 
                      (3, 8) : (2.5, 0.8),
                      (4, 2) : (1.5, 0.3), 
                      (4, 0) : (2.0, 0.8), 
                      (4, 9) : (2.5, 0.7), 
                      (5, 1) : (1.5, 0.3),
                      (5, 6) : (2.0, 0.7), 
                      (5, 5) : (1.0, 0.5), 
                      (6, 2) : (1.5, 0.8), 
                      (6, 7) : (2.1, 0.6), 
                      (6, 0) : (1.5, 0.2), 
                      (7, 0) : (1.5, 0.3), 
                      (7, 4) : (1.2, 0.9), 
                      (7, 7) : (0.9, 0.6), 
                      (8, 2) : (3.0, 0.6), 
                      (8, 5) : (2.0, 0.4), 
                      (9, 8) : (1.0, 0.7), 
                      (9, 1) : (2.3, 0.3),
                      (9, 3) : (1.6, 0.6)
                  },    
            
            
}

    
    for SIDE_N in POSSIBLE_SIDE_N:
        for seed in SEEDS:
            for conn_name, conn_config in conn.iteritems():
                for jitteri in range(len(JITTERS)):
                    filename = "ptdata.%d.%d.%d.%s.pickle" % (SIDE_N, seed, jitteri, 
                                                            conn_name)
                    yield None, filename, SIDE_N, seed, conn_name, conn_config, jitteri

@files(data_generator)
def create_data(inputfile, outputfile, SIDE_N, seed, conn_name, conn_config, jitteri):
    
    np.random.seed(seed)
    nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, 
                                                                         conn_config,
                                                                         JITTER=JITTERS[jitteri])
    
                
    conn_and_dist = np.zeros(connectivity.shape, 
                    dtype=[('link', np.uint8), 
                           ('distance', np.float32)])

    for ni, (ci, posi) in enumerate(nodes_with_class):
        for nj, (cj, posj) in enumerate(nodes_with_class):
            conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
            conn_and_dist[ni, nj]['distance'] = d(posi, posj)
    pickle.dump({'SIDE_N' : SIDE_N, 
                 'seed' : seed, 
                 'conn_name' : conn_name, 
                 'conn_config' : conn_config, 
                 'nodes' : nodes_with_class, 
                 'connectivity' : connectivity, 
                 'conn_and_dist' : conn_and_dist}, 
                open(outputfile, 'w'))

# def create_inference_ld():
#     INITS = SAMPLER_INITS
#     for x in data_generator():
#         filename = x[1]
#         otherargs = x[2:]
#         for seed in range(INITS):
#             outfilename = "%s.ld.%d.pickle" % (filename, init)
#             yield filename, outfilename, init


def inference_run_ld(latent_filename, 
                     data_filename, 
                     config_filename,  ITERS, seed):

    latent = from_bucket(latent_filename)
    data = from_bucket(data_filename)
    config = from_bucket(config_filename)

    SAVE_EVERY = 100
    chain_runner = irm.runner.Runner(latent, data, config, seed)

    scores = []
    times = []
    def logger(iter, model):
        print "Iter", iter
        scores.append(model.total_score())
        times.append(time.time())
    chain_runner.run_iters(ITERS, logger)
        
    return scores, chain_runner.get_state(), times


@transform(create_data, regex(r"(.+).pickle$"), 
            r"\1.samples." + ("%d" %(SAMPLER_ITERS)) + ".exp")
def create_rundata(infilename, outfilename):
    """
    Create the data to run the experiments
    """

    ITERS = SAMPLER_ITERS

    indata = pickle.load(open(infilename, 'r'))

    model_name= "LogisticDistance" 
    kernel_config = irm.runner.default_kernel_nonconj_config()
    kernel_config[0][1]['M'] = 30
    kernel_config_pt = [('parallel_tempering', {'temps' : [1.0, 2.0, 4.0, 8.0], 
                                                'subkernels' : kernel_config})]
    

    data = indata['conn_and_dist']
    nodes = indata['nodes']


    irm_latent, irm_data = irm.irmio.default_graph_init(data, model_name)

    HPS = {'mu_hp' : 1.0, 
           'lambda_hp' : 1.0, 
           'p_min' : 0.1, 
           'p_max' : 0.9}
    irm_latent['relations']['R1']['hps'] = HPS
    irm_latents = []
    kernel_configs = []
    for c in range(CHAINS_TO_RUN):
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        a = np.arange(irm_data['domains']['d1']['N']) % INITIAL_GROUP_NUM
        a = np.random.permutation(a)
    
        latent['domains']['d1']['assignment'] = a
        irm_latents.append(latent)
        kernel_configs.append(kernel_config_pt)

    # the ground truth one
    irm_latent_true = copy.deepcopy(irm_latent)
    irm_latent_true['domains']['d1']['assignment'] = nodes['class']
    irm_latents[0] = irm_latent_true
    
    filenames = {}
    data_filename = outfilename + ".data"

    pickle.dump(irm_data, open(data_filename, 'w'))
    to_bucket(data_filename)
    filenames['data'] = data_filename
    filenames['chains'] = {}
    # filenames
    for c in range(CHAINS_TO_RUN):
        s = outfilename + (".%d" % c)
        latent_filename = s + ".latent"
        config_filename = s + ".config"
        pickle.dump(irm_latents[c], open(latent_filename, 'w'))
        to_bucket(latent_filename)
        pickle.dump(kernel_configs[c], open(config_filename, 'w'))
        to_bucket(config_filename)

        filenames['chains'][c] = {'latent' : latent_filename, 
                                  'config' : config_filename}


    # jids = cloud.map(inference_run_ld, irm_latents,
    #                  [irm_data]*CHAINS_TO_RUN, 
    #                  kernel_configs,
    #                  [ITERS] * CHAINS_TO_RUN, 
    #                  range(CHAINS_TO_RUN), 
    #                  _env='connectivitymotif', 
    #                  _type='f2')

    # fixme save all inputs
    pickle.dump({'infile' : infilename, 
                 'hps' : HPS, 
                 'filenames' : filenames}, 
                open(outfilename, 'w'))

@transform(create_rundata, suffix(".exp"), ".exp.wait")
def start_inference(infilename, outfilename):

    ITERS = SAMPLER_ITERS

    indata = pickle.load(open(infilename, 'r'))
    filenames = indata['filenames']
    data_filename = filenames['data']
    latent_filenames = []
    config_filenames = []
    for chain, v in filenames['chains'].iteritems():
        latent_filenames.append(v['latent'])
        config_filenames.append(v['config'])
    
    jids = cloud.map(inference_run_ld, latent_filenames,
                     [data_filename]*CHAINS_TO_RUN, 
                     config_filenames,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     _env='connectivitymotif', 
                     _type='f2')

    # fixme save all inputs
    pickle.dump({'infile' : indata['infile'],
                 'filenames' : filenames,
                 'jids' : jids}, 
                open(outfilename, 'w'))

@transform(start_inference, regex(r"(.+).wait$"), 
            r"\1.pickle")
def get_inference(infilename, outfilename):
    d= pickle.load(open(infilename))
    
    chains = []
    # reorg on a per-seed basis
    for chain_data in cloud.iresult(d['jids'], ignore_errors=True):
        
        chains.append({'scores' : chain_data[0], 
                       'state' : chain_data[1], 
                       'times' : chain_data[2]})
        
        
    pickle.dump({'chains' : chains, 
                 'infile' : infilename}, 
                open(outfilename, 'w'))

GROUP_SIZE_THOLD = 0.97

def group_mass(group_sizes, thold):
    """
    What fraction of the groups account for thold of 
    the data? 

    1. sort the groups by size
    2. add until > thold
    """
    gs_a = np.sort(np.array(group_sizes))[::-1]
    gs_n = gs_a.astype(float) / np.sum(gs_a)
    
    tot = 0
    for i in range(len(gs_n)):
        tot += gs_n[i]
        if tot > thold:
            return i + 1
    return len(gs_n)
    



@transform(get_inference, regex(r"(.+).pickle$"),  
         [r"\1.scores.pdf", 
          r"\1.counts.pickle", 
          r"\1.params.png"])
def plot_collate(inputfile, (plot_outfile, counts_outfile, 
                             params_outfile)):
    filedata = pickle.load(open(inputfile))
    chains = filedata['chains']
    chains_infile = filedata['infile']
    print "CHAINS_INFILE=", chains_infile

    d = pickle.load(open(chains_infile, 'r'))
    d_infile = d['infile']
    print "D_INFILE=", d_infile
    original_data = pickle.load(open(d_infile, 'r'))
    nodes_with_class = original_data['nodes']
    true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_score = f.add_subplot(2, 3, 1)
    ax_groups =f.add_subplot(2, 3, 2) 
    ax_params = f.add_subplot(2, 3, 3)
    ax_score_groupcount = f.add_subplot(2, 3, 4)
    ax_purity_control = f.add_subplot(2, 3, 5)

    param_fig = pylab.figure(figsize=(2, CHAINN))
    #ax_each_chain = [param_fig.add_subplot(CHAINN, 1, i) for i in range(CHAINN)]
    ax_each_chain =  Grid(param_fig, 111, # similar to subplot(111)
                          nrows_ncols = (CHAINN, 1), 
                          axes_pad=0.1, # pad between axes in inch.
                           )

    groupcounts = []

    allscores = []

    params_mu = []
    params_lambda = []
    params_comp_size = []
    for di, d in enumerate(chains):
        this_iter_gc = []
        sample_latent = d['state']
        chain_params = {'size' : [], 
                        'lambda' : [], 
                        'mu' : []} 
        
        a = sample_latent['domains']['d1']['assignment']
        group_sizes = irm.util.count(a)
        gs = group_sizes.values()
        this_iter_gc.append(group_mass(gs, GROUP_SIZE_THOLD))
        components = sample_latent['relations']['R1']['ss']

        # this is fun and complex
        unique_gids = np.unique(a)
        for g1 in unique_gids:
            for g2 in  unique_gids:
                c = components[(g1, g2)]
                comp_size = group_sizes[g1] * group_sizes[g2]
                chain_params['lambda'].append(c['lambda'])
                chain_params['mu'].append(c['mu'])
                chain_params['size'].append(comp_size)
        params_mu += chain_params['mu']
        params_lambda += chain_params['lambda']
        params_comp_size += chain_params['size']

        groupcounts.append(len(group_sizes) )

        ax_each_chain[di].scatter(chain_params['mu'], 
                                  chain_params['lambda'], 
                                  alpha = 0.5, s = np.array(chain_params['size'])/200., 
                                  edgecolor='none')
        ax_each_chain[di].set_ylim(0, 0.4)
        ax_each_chain[di].tick_params(axis='both', which='minor',
                                      labelsize=6.0)
        ax_each_chain[di].tick_params(axis='both', which='major',
                                      labelsize=6.0)

    ax_params.scatter(params_mu, params_lambda, alpha=0.5, 
                      s= np.array(params_comp_size)/200., 
                      edgecolor='none')
    ax_params.grid(1)
    ax_params.set_xlabel('mu')
    ax_params.set_ylabel('lambda')

    mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])

    for di, d in enumerate(chains):
        subsamp = 4
        s = np.array(d['scores'])[::subsamp]
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

        allscores.append(d['scores'])
    sm = pylab.cm.ScalarMappable(cmap=mymap, 
                                 norm=pylab.normalize(vmin=1, vmax=5.0))
    sm._A = []
    ax_score.tick_params(axis='both', which='major', labelsize=6)
    ax_score.tick_params(axis='both', which='minor', labelsize=6)
    ax_score.set_xlabel('time (s)')
    ax_score.grid(1)
    all_s = np.hstack(allscores).flatten()
    #r = np.max(all_s) - np.min(all_s)
    #ax_score.set_ylim(np.min(all_s) + r*0.95, np.max(all_s)+r*0.05)
    
    bins = np.arange(1, 11)
    hist, _ = np.histogram(groupcounts, bins)

    ax_groups.bar(bins[:-1], hist)
    ax_groups.set_xlim(bins[0], bins[-1])

    scoregroup = []
    for di, d in enumerate(chains):
        assignment = d['state']['domains']['d1']['assignment']
        scores = d['scores']
        a = assignment
        group_sizes = irm.util.count(a)
        scoregroup.append((len(group_sizes), scores[-1]))

    scoregroup = np.array(scoregroup)

    jitter_counts = scoregroup[:, 0] + np.random.rand(len(scoregroup[:, 0])) * 0.2 -0.1
    ax_score_groupcount.scatter(jitter_counts, 
                                scoregroup[:, 1])

    
    ###### plot purity #######################
    ###
    tv = true_assignvect.argsort()
    tv_i = true_assignvect[tv]
    vals = [tv_i]
    # get the chain order 
    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    for di in chains_sorted_order: 
        d = chains[di] 
        sample_latent = d['state']
        a = np.array(sample_latent['domains']['d1']['assignment'])
        print "LEN=", len(np.unique(a))
        a_s = a.argsort(kind='heapsort')
        vals.append(true_assignvect[a_s])
    vals_img = np.vstack(vals)
    ax_purity_control.imshow(vals_img, interpolation='nearest')
    ax_purity_control.set_xticks([])
    ax_purity_control.set_yticks([])
    ax_purity_control.set_aspect(30)

    f.tight_layout()

    f.savefig(plot_outfile)

    param_fig.tight_layout()
    param_fig.savefig(params_outfile, dpi=300)
    

    pickle.dump({'counts' : groupcounts}, 
                open(counts_outfile, 'w'))

@transform(get_inference, regex(r"(.+).pickle$"),  
           [r"\1.latent.pdf", r"\1.sample.truth.pdf", 
            r"\1.sample.map.pdf"])
def plot_latent(inputfile, (latent_plot, true_sample, 
                            map_sample)):
    filedata = pickle.load(open(inputfile))
    chains = filedata['chains']
    chains_infile = filedata['infile']
    print "CHAINS_INFILE=", chains_infile

    d = pickle.load(open(chains_infile, 'r'))
    d_infile = d['infile']
    print "D_INFILE=", d_infile
    original_data = pickle.load(open(d_infile, 'r'))
    nodes_with_class = original_data['nodes']
    conn_and_dist = original_data['conn_and_dist']

    true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_purity_control = f.add_subplot(2, 2, 1)
    ax_z = f.add_subplot(2, 2, 2)

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
        sorted_assign_matrix.append(a)
    irm.plot_purity(ax_purity_control, true_assignvect, sorted_assign_matrix)

    # zmatrix
    av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
    z = irm.util.compute_zmatrix(av)    

    irm.plot.plot_zmatrix(ax_z, z)

    f.tight_layout()

    f.savefig(latent_plot)

    f = pylab.figure()
    plot_latent_sample(f, true_assignvect, conn_and_dist)
    f.savefig(true_sample)

    f = pylab.figure()
    ci = chains_sorted_order[0]
    cs = chains[ci]['state']
    plot_latent_sample(f, np.array(cs['domains']['d1']['assignment']),
                       conn_and_dist, 
                       model="LogisticDistance", 
                       comps = cs['relations']['R1']['ss'])

    f.savefig(map_sample)

def logistic(x, mu, lamb):
    return 1.0/(1 + np.exp((x - mu)/lamb))
    
def plot_latent_sample(fig, assign_vect, conn_and_dist, model=None, 
                       comps = None):


    #from mpl_toolkits.axes_grid1 import ImageGrid

    CLASSES = np.unique(assign_vect)
    CLASSN = len(CLASSES)
    
    # ax_grid = ImageGrid(fig, 111, # similar to subplot(111)
    #                     nrows_ncols = (CLASSN, CLASSN),
    #                     axes_pad = 0.1,
    #                     add_all=True,
    #                     label_mode = "L",
    #                 )
    for c1i, c1 in enumerate(CLASSES):
        for c2i, c2 in enumerate(CLASSES):
            ax = fig.add_subplot(CLASSN, CLASSN, c1i * CLASSN + c2i +1)
            nodes_1 = np.argwhere(assign_vect == c1).flatten()
            nodes_2 = np.argwhere(assign_vect == c2).flatten()
            conn_dist_hist = []
            noconn_dist_hist = []
            for n1 in nodes_1:
                for n2 in nodes_2:
                    d = conn_and_dist[n1, n2]['distance']
                    if conn_and_dist[n1, n2]['link']:
                        conn_dist_hist.append(d)
                    else:
                        noconn_dist_hist.append(d)

            bins = np.linspace(0, 10, 20)
            fine_bins = np.linspace(0, 10, 100)

            # compute prob as a function of distance for this class
            htrue, _ = np.histogram(conn_dist_hist, bins)
            print htrue
            hfalse, _ = np.histogram(noconn_dist_hist, bins)

            p = htrue.astype(float) / (hfalse + htrue)
            print p
            ax.plot(bins[:-1], p)
            ax.set_xlim(0, 10.0)
            ax.set_ylim(0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if model == "LogisticDistance":
                c = comps[(c1, c2)]
                print "c=", c
                ax.plot(fine_bins, 
                        logistic(fine_bins, c['mu'], c['lambda']), 
                        c='r') 
                ax.axvline(c['mu'], c='k')
            
pipeline_run([create_data, start_inference, get_inference, 
              plot_collate, plot_latent 
              #rand_collate
          ],
             multiprocess=4)
