from ruffus import *
import cPickle as pickle
import numpy as np

import irm
import irm.data
from matplotlib import pylab
import cloud
import rand

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

CHAINS_TO_RUN = 10
SAMPLER_ITERS = 1000
def data_generator():

    POSSIBLE_SIDE_N = [10]
    
    conn = {'twoclass': {(0, 1) : (2.0, 0.7), 
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
            
            
    }

    SEEDS = np.arange(6)
    
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

# def create_inference_ld():
#     INITS = SAMPLER_INITS
#     for x in data_generator():
#         filename = x[1]
#         otherargs = x[2:]
#         for seed in range(INITS):
#             outfilename = "%s.ld.%d.pickle" % (filename, init)
#             yield filename, outfilename, init


def inference_run_ld(irm_config, kernel_config,  ITERS, seed):

    rng = irm.RNG()
    np.random.seed(seed)

    model = irm.irmio.model_from_config(irm_config, init='crp', 
                                        rng=rng)

    rel = model.relations['R1']
    doms = [(model.domains['t1'], 0), (model.domains['t1'], 0)]
    scores = []
    states = []
    comps = []
    for i in range(ITERS):
        print "iteration", i
        irm.runner.do_inference(model, rng, kernel_config)
        a = model.domains['t1'].get_assignments()

        components = irm.model.get_components_in_relation(doms, rel)

        scores.append(model.total_score())
        states.append(a)
        comps.append(components)

    return scores, states, comps

@transform(create_data, regex(r"(.+).pickle$"), 
            r"\1.samples." + ("%d" %(SAMPLER_ITERS)) + ".wait")
def start_inference(infilename, outfilename):
    ITERS = SAMPLER_ITERS

    indata = pickle.load(open(infilename, 'r'))

    model_name= "LogisticDistance" 
    kernel_config = irm.runner.default_kernel_nonconj_config()
    kernel_config[0][1]['M'] = 30

    data = indata['conn_and_dist']


    irm_config = irm.irmio.default_graph_init(data, model_name)

    HPS = {'mu_hp' : 1.0, 
           'lambda_hp' : 1.0, 
           'p_min' : 0.1, 
           'p_max' : 0.9}
    irm_config['relations']['R1']['hps'] = HPS

    
    # scores, states, comps = inference_run_ld(irm_config, kernel_config, 
    #                                          ITERS, 
    #                                          seed)
    jids = cloud.map(inference_run_ld, [irm_config]*CHAINS_TO_RUN, 
                    [kernel_config]*CHAINS_TO_RUN, 
                    [ITERS] * CHAINS_TO_RUN, 
                    range(CHAINS_TO_RUN), 
                    _env='connectivitymotif', 
                     _type='f2')

    pickle.dump({'infile' : infilename, 
                 'irm_config' : irm_config, 
                 'hps' : HPS, 
                 'kernel_config' : kernel_config, 
                 'jids' : jids}, 
                open(outfilename, 'w'))

@transform(start_inference, regex(r"(.+).wait$"), 
            r"\1.pickle")
def get_inference(infilename, outfilename):
    d= pickle.load(open(infilename))
    res = cloud.result(d['jids'])
     
    chains = []
    # reorg on a per-seed basis
    for chain_data in cloud.iresult(d['jids']):
        chains.append({'scores' : chain_data[0], 
                       'states' : chain_data[1], 
                       'components' : chain_data[2]})
        
        
    pickle.dump({'chains' : chains, 
                 'infile' : infilename}, 
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
    

@transform(get_inference, regex(r"(.+).pickle$"),  
         [r"\1.scores.pdf", 
          r"\1.counts.pickle"])
def plot_collate(inputfile, (plot_outfile, counts_outfile)):
    print "INPUTFILE=", inputfile
    f = pylab.figure()
    ax_score = f.add_subplot(1, 2, 1)
    ax_groups =f.add_subplot(1, 2, 2) 
    groupcounts = []
    bins = np.arange(1, 7)
    filedata = pickle.load(open(inputfile))
    chains = filedata['chains']
    allscores = []
    meancounts = []
    for di, d in enumerate(chains):
        this_iter_gc = []
        assignments = d['states']
        for S in np.arange(BURN, len(assignments), SKIP):
            a = assignments[S]
            gs = irm.util.count(a).values()
            this_iter_gc.append(group_mass(gs, GROUP_SIZE_THOLD))
        meancounts.append(np.mean(this_iter_gc))
        groupcounts += this_iter_gc

    for di, d in enumerate(chains):
        gc_mean = meancounts[di]
        gc_min = np.min(groupcounts)
        gc_max = np.max(groupcounts)
        r = (gc_mean - gc_min) / (gc_max - gc_min)
        ax_score.plot(d['scores'], c=pylab.cm.jet(r))
        allscores.append(d['scores'])

    all_s = np.hstack(allscores).flatten()
    r = np.max(all_s) - np.min(all_s)
    ax_score.set_ylim(np.min(all_s) + r*0.95, np.max(all_s)+r*0.05)
    
    hist, _ = np.histogram(groupcounts, bins)
    print bins, hist
    ax_groups.bar(bins[:-1], hist)
    ax_groups.set_xlim(bins[0], bins[-1])
    f.savefig(plot_outfile)
    
    pickle.dump({'counts' : groupcounts}, 
                open(counts_outfile, 'w'))

@collate(get_inference, regex(r"(.+)\.\d+\.(.+)\.samples.+.pickle$"),  
         r"\1.\2.rand.pdf")
def rand_collate(inputfiles, rand_plot_filename):

    all_aris = []
    for inputfile in  inputfiles:
        filedata = pickle.load(open(inputfile))
        start_inference_file =  filedata['infile']
        start_inference_data = pickle.load(open(start_inference_file))
        orig_data = pickle.load(open(start_inference_data['infile'], 'r'))

        nodes = orig_data['nodes']
        orig_class = nodes['class']

        chains = filedata['chains']
        aris = []
        for di, d in enumerate(chains):
            assignments = d['states']
            for S in np.arange(BURN, len(assignments), SKIP):
                a = assignments[S]
                c_a = rand.canonicalize_assignment_vector(a)
                c_orig_class = rand.canonicalize_assignment_vector(orig_class)
                ari = rand.compute_adj_rand_index(c_a, c_orig_class)
                aris.append(ari)
        all_aris.append(aris)
    pylab.boxplot(all_aris)
    pylab.savefig(rand_plot_filename)

pipeline_run([create_data, start_inference, get_inference, 
              plot_collate, rand_collate],
             multiprocess=4)
