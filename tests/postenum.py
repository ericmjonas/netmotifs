import numpy as np
from ruffus import * 
import cPickle as pickle

from matplotlib import pylab
from copy import deepcopy

import irm
from irm import models
from irm import gibbs
from irm import util
from irm import irmio

from irm import relation

RELATION_CLASS = irm.Relation # relation.FastRelation

SAMPLE_SETS = 20
SAMPLES_N = 100
ITERS_PER_SAMPLE = 10
SEEDS = 4

def addelement(partlist, e):
    newpartlist = []
    for part in partlist:
        npart = part + [[e]]
        newpartlist += [npart]
        for i in xrange(len(part)):
            npart = deepcopy(part)
            npart[i] += [e]
            newpartlist += [npart]
    return newpartlist

def partition(n):
    if n == 0: return []
    partlist = [[[1]]]
    for i in xrange(2, n+1):
        partlist = addelement(partlist, i)
    return partlist

def canonicalize_assignment(assignments):
    """
    Canonicalize an assignment vector. this works as follows:
    largest group is group 0, 
    next largest is group 1, etc. 

    For two identically-sized groups, The lower one is
    the one with the smallest row
    
    """
    groups = {}
    for gi, g in enumerate(assignments):
        if g not in groups:
            groups[g] = []
        groups[g].append(gi)
    orig_ids = np.array(groups.keys())
    sizes = np.array([len(groups[k]) for k in orig_ids])

    unique_sizes = np.sort(np.unique(sizes))[::-1]
    out_assign = np.zeros(len(assignments), dtype=np.uint32)

    # unique_sizes is in big-to-small
    outpos = 0
    for size in unique_sizes:
        # get the groups of this size
        tgt_ids = orig_ids[sizes == size]
        minvals = [np.min(groups[tid]) for tid in tgt_ids]
        min_idx = np.argsort(minvals)
        for grp_id  in tgt_ids[min_idx]:
            out_assign[groups[grp_id]] = outpos
            outpos +=1 
    return out_assign

def part_to_assignvect(part, N):
    """
    partition [[1, 2, 3], [4, 5], [6]]
    to assignvect [0 0 0 1 1 2]
    """
    outvect = np.zeros(N, dtype=np.uint32)
    i = 0
    for pi, p in enumerate(part):
        outvect[np.array(p)-1] = pi
    return outvect

def enumerate_canonical_partitions(Nrows):
    parts = partition(Nrows)
    assignments = np.zeros((len(parts), Nrows), 
                            dtype = np.uint8)
    for pi, part in enumerate(parts):
        a = part_to_assignvect(part, Nrows)
        ca = canonicalize_assignment(a)
        assignments[pi] = ca
    return parts, assignments


def type_to_assign_vect(type_intf, av, rng=None):
    """
    for a given assignment vector, force the type into that format
    """
    assert len(av) == type_intf.entity_count()
    id_to_gid = {}

    for ei, a in enumerate(av):
        oldgroup = type_intf.remove_entity_from_group(ei)
        if type_intf.group_size(oldgroup) == 0:
            type_intf.delete_group(oldgroup)
        if a not in id_to_gid:
            new_gid = type_intf.create_group(rng)
            id_to_gid[a] = new_gid
        type_intf.add_entity_to_group(id_to_gid[a], ei)
    
def t1_t2_datasets():
    T1_N = 10
    T2_N = 8
    for seed in range(SEEDS):
        output_filename = "irm.t1xt2.%d.%d.%d.pickle" % (T1_N, T2_N, seed)
        yield None, output_filename, T1_N, T2_N, seed

@files(t1_t2_datasets)
def create_data_t1t2(inputfile, outputfile, T1_N, T2_N, seed):

    np.random.seed(seed)
    data = np.random.rand(T1_N, T2_N) > 0.5
    
    config = {'types' : {'t1' : {'hps' : 1.0, 
                                 'N' : T1_N}, 
                         't2' : {'hps' : 1.0, 
                                 'N' : T2_N}}, 
              'relations' : { 'R1' : {'relation' : ('t1', 't2'), 
                                      'model' : 'BetaBernoulli', 
                                      'hps' : {'alpha' : 1.0, 
                                               'beta' : 1.0}}}, 
              'data' : {'R1' : data}}

    pickle.dump(config, open(outputfile, 'w'))

def t1_t1_datasets():
    T1_N = 8
    for seed in range(SEEDS):
        output_filename = "irm.t1xt1.%d.%d.pickle" % (T1_N, seed)
        yield None, output_filename, T1_N,  seed

@files(t1_t1_datasets)
def create_data_t1t1(inputfile, outputfile, T1_N,  seed):

    np.random.seed(seed)
    data = np.random.rand(T1_N, T1_N) > 0.5
    
    config = {'types' : {'t1' : {'hps' : 1.0, 
                                 'N' : T1_N}},
              'relations' : { 'R1' : {'relation' : ('t1', 't1'), 
                                      'model' : 'BetaBernoulli', 
                                      'hps' : {'alpha' : 1.0, 
                                               'beta' : 1.0}}}, 
              'data' : {'R1' : data}}

    pickle.dump(config, open(outputfile, 'w'))


@transform([create_data_t1t2, create_data_t1t1], 
           suffix(".pickle"), ".samples.pickle")
def run_samples(infile, outfile):
    config = pickle.load(open(infile))

    rng = irm.RNG()

    irm_model = irmio.model_from_config(config, relation_class = RELATION_CLASS,
                                        rng = rng)

    # pick what the T1 is that we're going to use
    t1_name = sorted(irm_model.types.keys())[0]
    t1_obj = irm_model.types[t1_name]
    T1_N = t1_obj.entity_count()
    print "RUNNING FOR", t1_name

    # create dataset
    parts, assignments = enumerate_canonical_partitions(T1_N)
    
    scores = np.zeros(len(assignments))
    ca_to_pos = {}
    for ai, a in enumerate(assignments):
        type_to_assign_vect(t1_obj, a, rng)
        scores[ai]  = irm_model.total_score()
        ca_to_pos[tuple(a)] = ai

    true_probs = util.scores_to_prob(scores)

    bins = np.zeros((SAMPLE_SETS, len(assignments)), dtype=np.uint32)
    print "SAMPLING"
    for samp_set in range(SAMPLE_SETS):
        print "Samp_set", samp_set
        for s in range(SAMPLES_N):
            for i in range(ITERS_PER_SAMPLE):
                gibbs.gibbs_sample_type(t1_obj, rng)
            a = t1_obj.get_assignments()
            ca = canonicalize_assignment(a)
            pi = ca_to_pos[tuple(ca)]
            bins[samp_set, pi] += 1
    print "DONE"
    
    pickle.dump({'bins' : bins, 
                 'true_probs' : true_probs, 
                 'infile' : infile}, 
                open(outfile, 'w'))

@transform([create_data_t1t2, create_data_t1t1], 
           suffix(".pickle"), ".nonconj.samples.pickle")
def run_samples_nonconj(infile, outfile):
    config = pickle.load(open(infile))

    rng = irm.RNG()
    
    irm_model = irmio.model_from_config(config, relation_class=RELATION_CLASS, 
                                        rng = rng)
    config['relations']['R1']['model'] = "BetaBernoulliNonConj"

    irm_model_nonconj = irmio.model_from_config(config, relation_class=RELATION_CLASS, rng=rng)

    # pick what the T1 is that we're going to use
    t1_name = sorted(irm_model.types.keys())[0]
    t1_obj = irm_model.types[t1_name]
    T1_N = t1_obj.entity_count()
    print "RUNNING FOR", t1_name

    # create dataset
    parts, assignments = enumerate_canonical_partitions(T1_N)
    
    scores = np.zeros(len(assignments))
    ca_to_pos = {}
    for ai, a in enumerate(assignments):
        type_to_assign_vect(t1_obj, a, rng)
        scores[ai]  = irm_model.total_score()
        ca_to_pos[tuple(a)] = ai

    true_probs = util.scores_to_prob(scores)
    bins = np.zeros((SAMPLE_SETS, len(assignments)), dtype=np.uint32)
    print "SAMPLING"
    t1_obj_nonconj = irm_model.types[t1_name]
    
    M = 10
    for samp_set in range(SAMPLE_SETS):
        print "Samp_set", samp_set
        for s in range(SAMPLES_N):
            for i in range(ITERS_PER_SAMPLE):
                gibbs.gibbs_sample_type_nonconj(t1_obj_nonconj, M, rng)
            a = t1_obj_nonconj.get_assignments()
            ca = canonicalize_assignment(a)
            pi = ca_to_pos[tuple(ca)]
            bins[samp_set, pi] += 1
    print "DONE"
    
    pickle.dump({'bins' : bins, 
                 'true_probs' : true_probs, 
                 'infile' : infile}, 
                open(outfile, 'w'))

@collate([run_samples, run_samples_nonconj],
         regex(r"\.(.+).\d+.(.*)samples.pickle$"),  r'\1.\2kl.pdf')
def summarize(infiles, summary_file):
    PLOT_N = 80

    KLs = np.zeros((len(infiles), 1000))
    for infile_i, infile in enumerate(infiles):
        print "INFILE = ", infile
        data = pickle.load(open(infile, 'r'))

        bins = data['bins']
        true_probs = data['true_probs']
        SAMPLE_SETS = len(bins)
        print SAMPLE_SETS
        for sample_set_i in range(SAMPLE_SETS):
            flat_bins = np.sum(bins[:sample_set_i+1], axis=0)
            emp_hist = flat_bins.astype(np.float)/np.sum(flat_bins)
            probs_idx = np.argsort(true_probs)[::-1].flatten()
            prob_true = true_probs[probs_idx][:PLOT_N]
            prob_emp = emp_hist[probs_idx][:PLOT_N]
            kl = util.kl(prob_true + 0.001, prob_emp + 0.001)
            KLs[infile_i, sample_set_i] = kl
    KLs = KLs[:, :SAMPLE_SETS]
    
    pylab.figure()
    for row in KLs:
        pylab.plot(row)
    pylab.grid()
    pylab.savefig(summary_file)

if __name__ == "__main__":
    pipeline_run([create_data_t1t2, run_samples, run_samples_nonconj, 
                  summarize], multiprocess=4)
