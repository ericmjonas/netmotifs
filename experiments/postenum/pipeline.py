import numpy as np
from ruffus import * 
import cPickle as pickle

from matplotlib import pylab
from copy import deepcopy
import time


import irm
from irm import models
from irm import gibbs
from irm import util
from irm import irmio, runner

from irm import relation
import util as putil

SAMPLE_SETS = 20
SAMPLES_N = 400
ITERS_PER_SAMPLE = 10
# SAMPLE_SETS = 4
# SAMPLES_N = 100
# ITERS_PER_SAMPLE = 2
SEEDS = 1
MODEL_CLASSES = ['conj', 'nonconj']

tt_config_nonconj = [('parallel_tempering', {'temps' : [1.0, 2.0, 4.0, 8.0], 
                                               'subkernels' : runner.default_kernel_nonconj_config()})]

tt_config_conj = [('parallel_tempering', {'temps' : [1.0, 2.0, 4.0, 8.0], 
                                            'subkernels' : runner.default_kernel_config()})]


KERNEL_CONFIGS = {'default' : {'conj' : runner.default_kernel_config(), 
                               'nonconj' : runner.default_kernel_nonconj_config()}}

def t1_t2_datasets():
    T1_N = 8
    T2_N = 7
    for mc in MODEL_CLASSES:
        for seed in range(SEEDS):
            filename_base = "srm.t1xt2.%d.%d.%d.%s" % (T1_N, T2_N, seed, mc)
            latent_filename = filename_base + ".latent"
            data_filename = filename_base + ".data"
            yield None, (latent_filename, data_filename), T1_N, T2_N, seed, mc

@files(t1_t2_datasets)
def create_data_t1t2(inputfile, (latent_filename, data_filename), 
                     T1_N, T2_N, seed, model_class):

    np.random.seed(seed)
    data_mat = np.random.rand(T1_N, T2_N) > 0.5
    

    data = {'domains' : {'t1' : {'N' : T1_N},
                         't2' : {'N' : T2_N}}, 
            'relations' : {'R1' : {'relation' : ('t1', 't2'), 
                                    'model' : 'BetaBernoulli', 
                                    'data' : data_mat}}}
    if model_class == 'nonconj':
        data['relations']['R1']['model'] = "BetaBernoulliNonConj"
        
    latent =  { 'domains' : {'t1' : {'hps' : {'alpha' : 1.0}, 
                                     'assignment' : range(T1_N)}, 
                             't2' : {'hps' : {'alpha' : 1.0}, 
                                     'assignment' : range(T2_N)}},
                'relations' : { 'R1' : {'hps' : {'alpha' : 1.0, 
                                                 'beta' : 1.0}}}}
    pickle.dump(data, open(data_filename, 'w'))
    pickle.dump(latent, open(latent_filename, 'w'))

def t1_t1_datasets():
    T1_N = 4
    for mc in MODEL_CLASSES:
        for seed in range(SEEDS):
            for observed in [0, 1]:
                output_filename = "srm.t1xt1.%d.%d.%d.%s" % (T1_N, seed, 
                                                             observed, 
                                                             mc)
                latent_filename = output_filename + ".latent"
                data_filename = output_filename + ".data"
                yield None, (latent_filename, data_filename), T1_N,  seed, observed, mc

@files(t1_t1_datasets)
def create_data_t1t1(inputfile, (latent_filename, data_filename), T1_N,  
                     seed, observed, model_class):

    np.random.seed(seed)
    data = np.random.rand(T1_N, T1_N) > 0.5
    if observed:
        observed_data = (np.random.rand(T1_N, T1_N) > 0.5).astype(np.uint8)
    else:
        observed_data = None

    data = {'domains' : {'t1' : {'N' : T1_N}},
            'relations' : {'R1' : {'relation' : ('t1', 't1'), 
                                   'model' : 'BetaBernoulli', 
                                   'data' : data, 
                                   'observed' : observed_data}}}
    if model_class == 'nonconj':
        data['relations']['R1']['model'] = "BetaBernoulliNonConj"
        
    latent =  { 'domains' : {'t1' : {'hps' : {'alpha' : 1.0}, 
                                     'assignment' : range(T1_N)}}, 
                'relations' : { 'R1' : {'hps' : {'alpha' : 1.0, 
                                                 'beta' : 1.0}}}}
    pickle.dump(data, open(data_filename, 'w'))
    pickle.dump(latent, open(latent_filename, 'w'))


def dump_kernel_configs_params():
    for kc in KERNEL_CONFIGS:
        for key in KERNEL_CONFIGS[kc]:
            yield None, "%s.%s.config" % (kc, key), kc, key

@files(dump_kernel_configs_params)
def dump_kernel_configs(infile, outfile, kc, key):
    pickle.dump(KERNEL_CONFIGS[kc][key], open(outfile, 'w'))


def score_params():
    #for a in (list(t1_t2_datasets()) + list(t1_t1_datasets())):
    for a in (list(t1_t1_datasets())):
        latent_filename = a[1][0]
        data_filename = a[1][1]
        outfilename = latent_filename[:-(len("latent"))] + 'scores'
        if 'conj' in latent_filename:
            yield (latent_filename, data_filename), outfilename

#@follows(t1_t2_datasets)
@follows(t1_t1_datasets)
@files(score_params)
def score((latent_filename, data_filename), outfilename):
    latent = pickle.load(open(latent_filename, 'r'))
    data = pickle.load(open(data_filename, 'r'))
    
    rng = irm.RNG()
        
    irm_model = irmio.create_model_from_data(data, rng=rng)
    irmio.set_model_latent(irm_model, latent, rng)

    # now we go through and score every possible latent
    domain_names = sorted(data['domains'].keys())
    domain_sizes = [data['domains'][dn]['N'] for dn in domain_names]
    
    # create the dict
    candidate_partitions = list(putil.enumerate_possible_latents(domain_sizes))
    CANDIDATE_N = len(candidate_partitions)
    scores = {}
    for cpi, cp in enumerate(candidate_partitions):
        t1 = time.time()
        for di, av in enumerate(cp):
            domain_name = domain_names[di]
            latent['domains'][domain_name]['assignment'] = av
        irmio.set_model_latent(irm_model, latent, rng)
        scores[cp] = irm_model.total_score()
        t2 = time.time()
        delta = t2-t1
        if cpi % 1000 == 0:
            print "%s : %3.2f min left" % (outfilename, delta * (CANDIDATE_N - cpi)/60.)
    pickle.dump(scores, open(outfilename, 'w'))
    
def run_samples_params():
    #for a in (list(t1_t1_datasets()) + list(t1_t2_datasets())):
    for a in (list(t1_t1_datasets())): #  + list(t1_t2_datasets())):
        latent_filename = a[1][0]
        data_filename = a[1][1]
        for kc_name in KERNEL_CONFIGS:
            if "nonconj" in latent_filename:
                config_filename = "%s.nonconj.config" % kc_name
            else:
                config_filename = "%s.conj.config" % kc_name

            outfilename = latent_filename[:-(len("latent"))] + kc_name + ".samples" 
                                     
            yield (latent_filename, data_filename, config_filename), outfilename


@follows(t1_t2_datasets)
@follows(t1_t1_datasets)
@follows(dump_kernel_configs)
@follows(score)
@files(run_samples_params)
def run_samples((latent_filename, data_filename, config_filename), outfile):
    kernel_config = pickle.load(open(config_filename, 'r'))
    latent = pickle.load(open(latent_filename, 'r'))
    data = pickle.load(open(data_filename, 'r'))


    run = runner.Runner(latent, data, kernel_config)

    domain_names = sorted(data['domains'].keys())

    ss = []
    print "SAMPLING"
    for samp_set in range(SAMPLE_SETS):
        samp_set_items = {}
        print "Samp_set", samp_set
        for s in range(SAMPLES_N):
            t1 = time.time()
            run.run_iters(ITERS_PER_SAMPLE)
            a_s = []
            for dn in domain_names:
                a = putil.canonicalize_assignment(run.model.domains[dn].get_assignments())
                a = tuple(a)
                a_s.append(a)
            a_s = tuple(a_s)
            if a_s not in samp_set_items:
                samp_set_items[a_s] = 0
            samp_set_items[a_s] += 1
            t2 = time.time()
            delta = (t2-t1)
            approx_time_left = (SAMPLE_SETS-samp_set)*delta*SAMPLES_N
            print "%s : roughly %3.2f min left" %(outfile, approx_time_left/60.)

        ss.append(samp_set_items)
    print "DONE"
    
    pickle.dump({'samp_set_items' : ss, 
                 #'true_probs' : true_probs, 
                 #'infile' : infile}, 
                 },
                open(outfile, 'w'))


@collate(run_samples,
         regex(r"(.+\..+)\..+\.(.+)\.samples$"),  [r'\1-\2.kl.pdf', 
                                                   r'\1-\2.dists.pdf'], r"\1")
def summarize(infiles, (kl_summary_file, dist_summary_file) , x):
    print "infile =", infiles, x
    scores_filename = x + ".conj.scores"

    scores = pickle.load(open(scores_filename))

    # now create the vector we will use
    score_vect = np.zeros(len(scores))
    latent_to_pos = {}
    pos = 0
    for l in scores.keys():
        latent_to_pos[l] = pos
        score_vect[pos] = scores[l]
        pos += 1
    probs = util.scores_to_prob(score_vect)

    prob_idx = np.argsort(probs)[::-1]

    probs = probs[prob_idx]

    PLOT_N = 10

    # compute the probabilities

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(probs[:PLOT_N], c='k')
    for infile_i, infile in enumerate(infiles):
        print "INFILE = ", infile
        samp_set_items = pickle.load(open(infile, 'r'))['samp_set_items']

        SAMPLE_SETS = len(samp_set_items)
        counts = np.zeros((SAMPLE_SETS, len(probs)))
        for sample_set_i in range(SAMPLE_SETS):
            ss = samp_set_items[sample_set_i]
            for k, v in ss.iteritems():
                k_i = latent_to_pos[k]
                counts[sample_set_i, k_i] += v
        a = counts.sum(axis=0).astype(float)
        a = a / np.sum(a)
        
        if 'non' in infile:
            label = 'nonconj'
            color = 'r'
        else:
            label = 'conj' 
            color = 'b'
        ax.scatter(range(PLOT_N), a[prob_idx][:PLOT_N], 
                   alpha=0.5, label = label, c=color)
    ax.legend()
    f.savefig(dist_summary_file)
        
            # flat_bins = np.sum(bins[:sample_set_i+1], axis=0)
            # emp_hist = flat_bins.astype(np.float)/np.sum(flat_bins)
            # probs_idx = np.argsort(true_probs)[::-1].flatten()
            # prob_true = true_probs[probs_idx][:KL_N]
            # prob_emp = emp_hist[probs_idx][:KL_N]
            # kl = util.kl(prob_true + 0.001, prob_emp + 0.001)
            # KLs[infile_i, sample_set_i] = kl

#     KLs = KLs[:, :SAMPLE_SETS]
    
#     pylab.figure()

    
#     for row in KLs:
#         pylab.plot(row)
#     pylab.grid()
#     pylab.savefig(kl_summary_file)

#     PLOT_N = 40
#     pylab.figure()
#     for infile_i, infile in enumerate(infiles):
#         pylab.subplot(len(infiles), 1, infile_i + 1)

#         data = pickle.load(open(infile, 'r'))

#         bins = data['bins']
#         true_probs = data['true_probs']
#         SAMPLE_SETS = len(bins)
#         sample_set_i = SAMPLE_SETS - 1
#         flat_bins = np.sum(bins[:sample_set_i+1], axis=0)
#         emp_hist = flat_bins.astype(np.float)/np.sum(flat_bins)
#         probs_idx = np.argsort(true_probs)[::-1].flatten()
#         prob_true = true_probs[probs_idx][:PLOT_N]
#         prob_emp = emp_hist[probs_idx][:PLOT_N]
#         pylab.plot(prob_true)
#         pylab.scatter(range(PLOT_N), prob_emp)
#         pylab.grid()
#     pylab.savefig(dist_summary_file)

if __name__ == "__main__":
    pipeline_run([#create_data_t1t2, 
                  create_data_t1t1, 
                  dump_kernel_configs, score, 
                  run_samples, summarize])
                  
