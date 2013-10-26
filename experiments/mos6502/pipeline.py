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

import cloud

BUCKET_BASE="srm/experiments/mos6502"


EXPERIMENTS = [('mos6502.all.decode.bb', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.all.xysregs.bb', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.all.lower.bb', 'fixed_20_200', 'anneal_slow_400'), 
               #('mos6502.all.all.bb', 'fixed_20_200', 'anneal_slow_400'), 

               ('mos6502.dir.decode.bb', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.dir.xysregs.bb', 'fixed_20_200', 'anneal_slow_400'),
               ('mos6502.dir.lower.bb', 'fixed_20_200', 'anneal_slow_400'),
               #('mos6502.dir.all.bb', 'fixed_20_200', 'anneal_slow_400'), 
               
               ('mos6502.all.decode.ld', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.all.xysregs.ld', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.all.lower.ld', 'fixed_20_200', 'anneal_slow_400'), 
               #('mos6502.all.all.ld', 'fixed_20_200', 'anneal_slow_400'), 

               ('mos6502.dir.decode.ld', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.dir.xysregs.ld', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.dir.lower.ld', 'fixed_20_200', 'anneal_slow_400'), 
               #('mos6502.dir.all.ld', 'fixed_20_200', 'anneal_slow_400'), 

               ('mos6502.count.decode.edp', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.count.xysregs.edp', 'fixed_20_200', 'anneal_slow_400'), 
               ('mos6502.count.lower.edp', 'fixed_20_200', 'anneal_slow_400'), 

               # ('mos6502.all.bb', 'fixed_20_200', 'default_100'), 
               #('mos6502.all.ld', 'fixed_20_200', 'anneal_slow_400'), 
           ]

INIT_CONFIGS = {'fixed_20_200' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
                'fixed_2_40' : {'N' : 2, 
                                'config' : {'type' : 'fixed', 
                                              'group_num' : 40}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}}
                
WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)
                

slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 128.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300

slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistance'] = irm.gridgibbshps.default_grid_logistic_distance(500)


KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},

                  }


def to_bucket(filename):
    cloud.bucket.sync_to_cloud(filename, os.path.join(BUCKET_BASE, filename))

def from_bucket(filename):
    return pickle.load(cloud.bucket.getf(os.path.join(BUCKET_BASE, filename)))


@transform('[ad]*.region.pickle', regex("(.+)\.region.pickle"), r'%s/mos6502.\1.data.pickle' % WORKING_DIR)
def data_mos6502_region(infile, all_file):
    data = pickle.load(open(infile, 'r'))

    
    pickle.dump({'dist_matrix' : data['adj_mat'], 
                 'infile' : infile}, open(all_file, 'w'))

@transform('count*.region.pickle', regex("(.+)\.region.pickle"), r'%s/mos6502.\1.data.pickle' % WORKING_DIR)
def data_mos6502_region_count(infile, all_file):
    data = pickle.load(open(infile, 'r'))

    
    pickle.dump({'dist_matrix' : data['adj_mat'], 
                 'infile' : infile}, open(all_file, 'w'))

@transform(data_mos6502_region, suffix(".data.pickle"), [".ld.data", ".ld.latent", ".ld.meta"])
def create_latents_ld(infile, 
                      (data_filename, latent_filename, meta_filename)):
    print "INPUT FILE IS", infile
    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "LogisticDistance" 

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'mu_hp' : 1000., 
           'lambda_hp' : 1000., 
           'p_min' : 0.0001, 
           'p_max' : 0.95}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))


@transform(data_mos6502_region, suffix(".data.pickle"), [".bb.data", ".bb.latent", ".bb.meta"])
def create_latents_bb(infile, 
                      (data_filename, latent_filename, meta_filename)):

    d = pickle.load(open(infile, 'r'))
    conn = d['dist_matrix']['link']
    
    model_name= "BetaBernoulli"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn, model_name)

    HPS = {'alpha' : 0.1, 
           'beta' : 0.1}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, open(meta_filename, 'w'))


@transform(data_mos6502_region_count, suffix(".data.pickle"), [".edp.data", ".edp.latent", ".edp.meta"])
def create_latents_edp(infile, 
                      (data_filename, latent_filename, meta_filename)):
    print "INPUT FILE IS", infile
    d = pickle.load(open(infile, 'r'))
    conn_and_dist = d['dist_matrix']
    
    model_name= "ExponentialDistancePoisson"

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_and_dist, model_name)

    HPS = {'mu_hp' : 1000., 
           'rate_scale_hp' : 1000.}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))


def create_init(latent_filename, out_filenames, 
                init= None):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    
    irm_latents = []

    for c, out_f in enumerate(out_filenames):
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

        # delete the suffstats
        if 'ss' in latent['relations']['R1']:
            del latent['relations']['R1']['ss']

        pickle.dump(latent, open(out_f, 'w'))


def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


            

@follows(create_latents_ld, create_latents_bb, create_latents_edp)
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
    
    jids = cloud.map(irm.experiments.inference_run, inits, 
                     [data_filename]*CHAINS_TO_RUN, 
                     [kernel_config]*CHAINS_TO_RUN,
                     [ITERS] * CHAINS_TO_RUN, 
                     range(CHAINS_TO_RUN), 
                     [BUCKET_BASE]*CHAINS_TO_RUN,
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

# @transform(get_results, suffix(".samples"), [".latent.pdf"])
# def plot_latent(exp_results, (plot_latent_filename, )):
#     sample_d = pickle.load(open(exp_results))
#     chains = sample_d['chains']
    
#     exp = sample_d['exp']
#     data_filename = exp['data_filename']
#     data = pickle.load(open(data_filename))
#     data_basename, _ = os.path.splitext(data_filename)
#     meta = pickle.load(open(data_basename + ".meta"))


#     # nodes_with_class = meta['nodes']
#     # conn_and_dist = meta['conn_and_dist']

#     # true_assignvect = nodes_with_class['class']

#     chains = [c for c in chains if type(c['scores']) != int]
#     CHAINN = len(chains)

#     f = pylab.figure(figsize= (12, 8))
#     ax_purity_control = f.add_subplot(2, 2, 1)
#     ax_z = f.add_subplot(2, 2, 2)
#     ax_score = f.add_subplot(2, 2, 3)
    
#     # ###### plot purity #######################
#     # ###
#     # tv = true_assignvect.argsort()
#     # tv_i = true_assignvect[tv]
#     # vals = [tv_i]
#     # # get the chain order 
#     # chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
#     # sorted_assign_matrix = []
#     # for di in chains_sorted_order: 
#     #     d = chains[di] 
#     #     sample_latent = d['state']
#     #     a = np.array(sample_latent['domains']['d1']['assignment'])
#     #     print "di=%d unique classes:"  % di, np.unique(a)
#     #     sorted_assign_matrix.append(a)
#     # irm.plot.plot_purity(ax_purity_control, true_assignvect, sorted_assign_matrix)

#     ###### zmatrix
#     av = [np.array(d['state']['domains']['d1']['assignment']) for d in chains]
#     z = irm.util.compute_zmatrix(av)    

#     irm.plot.plot_zmatrix(ax_z, z)

#     ### Plot scores
#     for di, d in enumerate(chains):
#         subsamp = 4
#         s = np.array(d['scores'])[::subsamp]
#         t = np.array(d['times'])[::subsamp] - d['times'][0]
#         ax_score.plot(t, s, alpha=0.7, c='k')

#     ax_score.tick_params(axis='both', which='major', labelsize=6)
#     ax_score.tick_params(axis='both', which='minor', labelsize=6)
#     ax_score.set_xlabel('time (s)')
#     ax_score.grid(1)
    
#     f.tight_layout()

#     f.savefig(plot_latent_filename)

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
    #cell_types = orig_data['types'][:len(conn)]

    # nodes_with_class = meta['nodes']
    # conn_and_dist = meta['conn_and_dist']

    # true_assignvect = nodes_with_class['class']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    f = pylab.figure(figsize= (12, 8))
    ax_purity_control = f.add_subplot(2, 2, 1)
    ax_z = f.add_subplot(2, 2, 2)
    ax_score = f.add_subplot(2, 2, 3)
    
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
    f = pylab.figure(figsize=(20, 4))

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
           [(".%d.latent.pdf" % d, ".%d.latent.pickle" % d)  for d in range(2)])
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

    d = pickle.load(open(meta_infile, 'r'))
    dist_matrix = d['dist_matrix']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)
    print "CHAINN=", CHAINN, "out_filenames=", out_filenames
    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (latent_plot_fname, 
                    latent_pickle_fname) in enumerate(out_filenames):
        print "plotting chain", chain_pos

        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']

        model = data['relations']['R1']['model']
        irm.experiments.plot_latent(sample_latent, dist_matrix, latent_plot_fname, 
                                    model=model, PLOT_MAX_DIST=10000)
    
        pickle.dump(sample_latent, open(latent_pickle_fname, 'w'))


@transform(get_results, suffix(".samples"), 
           [(".%d.circos.png" % d,)  for d in range(1)])
def plot_best_circos(exp_results, 
                     out_filenames):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']

    d = pickle.load(open(meta_infile, 'r'))
    dist_matrix = d['dist_matrix']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    for chain_pos, (latent_plot_fname, ) in enumerate(out_filenames):


        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        a = sample_latent['domains']['d1']['assignment']

        cp = irm.plots.circos.CircosPlot(a)
        # cp.set_entity_labels(canonical_neuron_ordering, label_size="15p")

        links = []
        if data['relations']['R1']['model'] in ["BetaBernoulli", "GammaPoisson"]:
            link_data = data['relations']['R1']['data']
        else:
            link_data = data['relations']['R1']['data']['link']
        for r in range(len(a)):
            for c in range(len(a)):
                if link_data[r, c]  > 0:
                    links.append((r, c))
        links = np.array(links)
        LINK_MAX = 2500
        if len(links) > LINK_MAX:
            links = links[np.random.permutation(len(links))[:(LINK_MAX-1)]]
        cp.set_entity_links(links)

        irm.plots.circos.write(cp, latent_plot_fname)



pipeline_run([data_mos6502_region, 
              create_inits, get_results, plot_best_latent, plot_scores_z, 
              plot_hypers, plot_best_circos])
                        
