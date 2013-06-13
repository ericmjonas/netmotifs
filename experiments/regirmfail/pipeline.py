from ruffus import *
import cPickle as pickle
import numpy as np

import irm
import irm.data
from matplotlib import pylab

def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

"""
for N connection classes 
for M seeds
    generate the data
    run the regular IRM on it
    run our iIRM on it with 10 different rng inits, 10 different seeds

"""

SAMPLER_INITS = 8
SAMPLER_ITERS = 1000
def data_generator():

    POSSIBLE_SIDE_N = [10]
    
    conn = {'twoclass': {(0, 1) : (2.0, 0.7), 
                         (1, 0) : (3.0, 0.8)},
            'oneclass': {(0, 0) : (2.0, 0.8)}}
    SEEDS = np.arange(4)
    
    for SIDE_N in POSSIBLE_SIDE_N:
        for seed in SEEDS:
            for conn_name, conn_config in conn.iteritems():
                filename = "data.%d.%d.%s.pickle" % (SIDE_N, seed, conn_name)
                yield None, filename, SIDE_N, seed, conn_name, conn_config

@files(data_generator)
def create_data(inputfile, outputfile, SIDE_N, seed, conn_name, conn_config):
    
    np.random.seed(seed)
    nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, conn_config)
    
                
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

def create_inference():
    INITS = SAMPLER_INITS
    for x in data_generator():
        filename = x[1]
        otherargs = x[2:]
        for init in range(INITS):
            outfilename = "%s.bbconj.%d.pickle" % (filename, init)
            yield filename, outfilename, init

@follows(create_data)
@files(create_inference)
def run_bbconj(infilename, outfilename, seed):
    ITERS = SAMPLER_ITERS

    np.random.seed(seed)

    indata = pickle.load(open(infilename, 'r'))

    model_name= "BetaBernoulli" 
    kc = irm.runner.default_kernel_config()

    data = indata['connectivity']
    
    irm_config = irm.irmio.default_graph_init(data, model_name)

    rng = irm.RNG()
    model = irm.irmio.model_from_config(irm_config, init='crp', 
                                        rng=rng)


    scores = []
    states = []
    comps = []
    for i in range(ITERS):
        print "iteration", i
        irm.runner.do_inference(model, rng, kc)
        a = model.domains['t1'].get_assignments()

        scores.append(model.total_score())
        states.append(a)

    pickle.dump({'scores' : scores, 
                 'states' : states, 
                 'infile' : infilename}, 
                open(outfilename, 'w'))

def create_inference_ld():
    INITS = SAMPLER_INITS
    for x in data_generator():
        filename = x[1]
        otherargs = x[2:]
        for init in range(INITS):
            outfilename = "%s.ld.%d.pickle" % (filename, init)
            yield filename, outfilename, init

@follows(create_data)
@files(create_inference_ld)
def run_ld(infilename, outfilename, seed):
    ITERS = SAMPLER_ITERS

    np.random.seed(seed)

    indata = pickle.load(open(infilename, 'r'))

    model_name= "LogisticDistance" 
    kc = irm.runner.default_kernel_nonconj_config()
    kc[0][1]['M'] = 30

    data = indata['conn_and_dist']


    irm_config = irm.irmio.default_graph_init(data, model_name)

    HPS = {'mu_hp' : 1.0, 
           'lambda_hp' : 1.0, 
           'p_min' : 0.1, 
           'p_max' : 0.9}
    irm_config['relations']['R1']['hps'] = HPS

    rng = irm.RNG()

    model = irm.irmio.model_from_config(irm_config, init='crp', 
                                        rng=rng)

    rel = model.relations['R1']
    doms = [(model.domains['t1'], 0), (model.domains['t1'], 0)]
    scores = []
    states = []
    comps = []
    for i in range(ITERS):
        print "iteration", i
        irm.runner.do_inference(model, rng, kc)
        a = model.domains['t1'].get_assignments()

        components = irm.model.get_components_in_relation(doms, rel)

        scores.append(model.total_score())
        states.append(a)
        comps.append(components)

    pickle.dump({'scores' : scores, 
                 'states' : states, 
                 'components' :  components, 
                 'infile' : infilename, 
                 'hps' : HPS}, 
                open(outfilename, 'w'))

GROUP_SIZE_THOLD = 0.85
SKIP = 100
BURN = 800
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
    

@collate(run_ld, regex(r"(.+)\.ld\.\d.pickle$"),  [r"\1.ld.scores.pdf", 
                                                   r"\1.ld.counts.pickle"])
def ld_collate(inputfiles, (plot_outfile, counts_outfile)):
    f = pylab.figure()
    ax_score = f.add_subplot(1, 2, 1)
    ax_groups =f.add_subplot(1, 2, 2) 
    groupcounts = []
    bins = np.arange(1, 7)
    for infile in inputfiles:
        d = pickle.load(open(infile))
        ax_score.plot(d['scores'])
        assignments = d['states']
        for S in np.arange(BURN, len(assignments), SKIP):
            a = assignments[S]
            gs = irm.util.count(a).values()
            groupcounts.append(group_mass(gs, GROUP_SIZE_THOLD))

    hist, _ = np.histogram(groupcounts, bins)
    print bins, hist
    ax_groups.bar(bins[:-1], hist)
    ax_groups.set_xlim(bins[0], bins[-1])
    f.savefig(plot_outfile)
    
    pickle.dump({'counts' : groupcounts}, 
                open(counts_outfile, 'w'))

@collate(run_bbconj, regex(r"(.+)\.bbconj\.\d.pickle$"),  [r"\1.bbconj.scores.pdf", 
                                                           r"\1.bbconj.scores.pickle"])
def bbconj_collate(inputfiles, (plot_outfile, counts_outfile)):
    f = pylab.figure()
    ax_score = f.add_subplot(1, 2, 1)
    ax_groups =f.add_subplot(1, 2, 2) 
    groupcounts = []

    bins = np.arange(1, 20)
    for infile in inputfiles:
        d = pickle.load(open(infile))
        ax_score.plot(d['scores'])
        assignments = d['states']
        for S in np.arange(BURN, len(assignments), SKIP):
            a = assignments[S]
            gs = irm.util.count(a).values()
            groupcounts.append(group_mass(gs, GROUP_SIZE_THOLD))

    print groupcounts
    hist, _ = np.histogram(groupcounts, bins)
    print bins, hist
    ax_groups.bar(bins[:-1], hist)
    ax_groups.set_xlim(bins[0], bins[-1])
    f.savefig(plot_outfile)
        
    pickle.dump({'counts' : groupcounts}, 
                open(counts_outfile, 'w'))

@merge([bbconj_collate, ld_collate], "comparison.pdf")
def all_plot(infiles, outfile):
    oneclass = {'ld' : [], 
                'bb' : []}
    twoclass = {'ld' : [], 
                'bb' : []}
    for _, pickle_filename in infiles:
        d = pickle.load(open(pickle_filename))

        if 'oneclass' in pickle_filename:
            if 'ld' in pickle_filename:
                oneclass['ld'] += d['counts']
            else:
                oneclass['bb'] += d['counts']
        elif 'twoclass' in pickle_filename:
            if 'ld' in pickle_filename:
                twoclass['ld'] += d['counts']
            else:
                twoclass['bb'] += d['counts']
    pylab.figure()
    bins = np.arange(0, 16)

    def non_suck_bar(ax, d):
        hist, _ = np.histogram(d, bins)
        ax.bar(bins[:-1]-0.5, hist, width=0.8)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(bins)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    ax = pylab.subplot(2, 2, 1)
    ax.axvline(1, linestyle='--', c='k')
    non_suck_bar(ax, oneclass['bb'])
    ax.set_title("One true latent class")
    ax.set_ylabel("Beta-Bernoulli Stochastic Blockmodel", 
                  fontsize=10)

    ax = pylab.subplot(2, 2, 3)
    ax.axvline(1, linestyle='--', c='k')
    non_suck_bar(ax, oneclass['ld'])
    ax.set_ylabel("Logistic Spatial-Relational Model", fontsize=10)

    ax = pylab.subplot(2, 2, 2)
    ax.axvline(2, linestyle='--', c='k')
    non_suck_bar(ax, twoclass['bb'])
    ax.set_title("Two true latent class")

    ax = pylab.subplot(2, 2, 4)
    ax.axvline(2, linestyle='--', c='k')
    non_suck_bar(ax, twoclass['ld'])


    pylab.tight_layout()
    pylab.savefig(outfile)
pipeline_run([create_data, run_bbconj, run_ld, ld_collate, 
              bbconj_collate, all_plot], 
             multiprocess=4)
