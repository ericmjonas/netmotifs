from ruffus import *
import pandas
from matplotlib import pylab
import cPickle as pickle
import preprocess
import numpy as np
import irm
import irm.data
import glob
import os
import copy
import cloud


np.random.seed(0)

@files(['apps.pickle', 'users.pickle', 'jobs.%d.pickle' % preprocess.WINDOW_N,
        'zipcodes.pickle'], 
       'data.pickle')
def create_data((apps_filename, users_filename, jobs_filename, 
                 zipcodes_filename), output_filename):
    apps = pickle.load(open(apps_filename, 'r'))['apps']
    users = pickle.load(open(users_filename, 'r'))['users']
    jobs = pickle.load(open(jobs_filename, 'r'))['jobs']

    zip_codes = pickle.load(open(zipcodes_filename, 'r'))['all']

    apps_2 = apps[apps['WindowID'] == preprocess.WINDOW_N]

    vc_users = apps_2['UserID'].value_counts()

    USER_N = 500

    APPLY_THOLD_MIN = 10
    # randomly select users who applied for at least APPLY_THOLD_MIN jobs
    user_subset = vc_users[vc_users > APPLY_THOLD_MIN]
    user_subset = user_subset.ix[np.random.choice(user_subset.index.values, USER_N)]
    user_subset = users.ix[user_subset.index.values]

    JOB_N = 500
    # of the jobs that they all applied for, take the JOB_N most popular

    job_subset = apps_2[apps_2['UserID'].isin(user_subset.index.values)]
    js_c = job_subset['JobID'].value_counts()
    job_ids = js_c[:JOB_N].index.values
    
    job_subset = jobs.ix[job_ids]

    job_subset = job_subset.join(zip_codes, on='Zip5')
    job_subset = job_subset.dropna(subset=['latitude', 'longitude'])

    user_subset = user_subset.join(zip_codes, on='ZipCode')
    user_subset = user_subset.dropna(subset=['latitude', 'longitude'])
    
    pickle.dump({'jobs' : job_subset, 
                 'users' : user_subset}, 
                open(output_filename, 'w'))

@follows(create_data)
@files(['data.pickle', 'apps.pickle'], 'dataset.pickle')
def dataset_create((data_filename, apps_filename), dataset_filename):
    data_subset = pickle.load(open(data_filename, 'r'))
    jobs = data_subset['jobs']
    users = data_subset['users']
    
    apps = pickle.load(open(apps_filename, 'r'))['apps']
    apps_2 = apps[apps['WindowID'] == preprocess.WINDOW_N]

    apps_subset = apps_2[apps_2['UserID'].isin(users.index.values)]
    apps_subset = apps_subset[apps_subset['JobID'].isin(jobs.index.values)]

    USERS_N = len(users)
    JOBS_N = len(jobs)

    
    # create the distance matrix
    conn = np.zeros((USERS_N, JOBS_N), 
                    dtype=[('link', np.uint8), 
                           ('distance', np.float32)])
    u_id_lookup = {id : pos for pos, id in enumerate(users.index.values)}
    j_id_lookup = {id : pos for pos, id in enumerate(jobs.index.values)}

    # first compute distance matrix
    for ui, u_row in users.iterrows():
        u_x = u_row['longitude']
        u_y = u_row['latitude']
        for ji, j_row in jobs.iterrows():
            j_x = j_row['longitude']
            j_y = j_row['latitude']
            
            d = np.sqrt((j_x - u_x)**2 + (j_y - u_y)**2)
            conn[u_id_lookup[ui], j_id_lookup[ji]]['distance'] = d
    for ai, a_row in apps_subset.iterrows():
        ui = u_id_lookup[a_row['UserID']]
        ji = j_id_lookup[a_row['JobID']]
        conn[ui, ji]['link'] = True
    pickle.dump({'conn' : conn, 
                 'u_id_lookup' : u_id_lookup, 
                 'j_id_lookup' : j_id_lookup, 
                 'apps_subset' : apps_subset, 
                 'users' : users, 
                 'jobs' : jobs}, 
                open(dataset_filename, 'w'))

@follows(dataset_create)
@files('dataset.pickle', 'output.pdf')
def dataset_debug(infile, outfile):
    data = pickle.load(open(infile, 'r'))
    f = pylab.figure()
    ax = f.add_subplot(2, 1, 1)
    ax.hist(data['conn']['distance'].flatten(), bins=20)
    ax = f.add_subplot(2, 1, 2)
    distances = []
    fd = data['conn'].flatten()
    for i in fd:
        if i['link']:
             distances.append(i['distance'])
    ax.hist(distances, bins=np.linspace(0, 10, 100))
    f.savefig(outfile)

BUCKET_BASE="srm/experiments/celegans"

WORKING_DIR = "data"

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300

def generate_ld_hypers():
    space_vals =  irm.util.logspace(5.0, 80.0, 20)
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



EXPERIMENTS = [('jobs.bb', 'fixed_10_100', 'nc_10'),
               ('jobs.bb', 'fixed_100_200', 'nc_100'), 
               ('jobs.ld', 'fixed_100_200', 'nc_100'),
               ('jobs.bb', 'fixed_100_200', 'nc_1000'), 
               ('jobs.ld', 'fixed_100_200', 'nc_1000'),
               ('jobs.bb', 'fixed_100_200', 'anneal_slow_400'), 
               ('jobs.ld', 'fixed_100_200', 'anneal_slow_400')]
    

INIT_CONFIGS = {'fixed_10_100' : {'N' : 10, 
                                 'config' : {'type' : 'fixed', 
                                             'group_num' : 100}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
}

default_nonconj = irm.runner.default_kernel_nonconj_config()
KERNEL_CONFIGS = {
                  'nc_10' : {'ITERS' : 10, 
                             'kernels' : default_nonconj},
                  'nc_100' : {'ITERS' : 100, 
                             'kernels' : default_nonconj},
                  'nc_1000' : {'ITERS' : 1000, 
                             'kernels' : default_nonconj},
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},
                  }

def create_jobs_latent(connectivity, model_name):
    USER_N, JOB_N = connectivity.shape

    latent = {'domains' : {'users' : {'hps' : {'alpha' : 1.0},
                                   'assignment' : np.arange(USER_N) % 50} , 
                           'jobs' : {'hps' : {'alpha' : 1.0}, 
                                     'assignment' : np.arange(JOB_N) % 50}}, 
              
              'relations' : { 'R1' : {'hps' : {'alpha' : 1.0, 
                                               'beta' : 1.0}}}}

    data = {'domains' : {'users' : {  'N' : USER_N}, 
                         'jobs' : {'N' : JOB_N}}, 
            'relations' : { 'R1' : {'relation' : ('users', 'jobs'), 
                                    'model' : model_name, 
                                    'data' : connectivity}}}
    return latent, data

@follows(dataset_create)
@files('dataset.pickle', ['jobs.bb.data', 'jobs.bb.latent', 'jobs.bb.meta'])
def create_latents_bb(infile, (data_filename, latent_filename, meta_filename)):
    d = pickle.load(open(infile, 'r'))
    conn_matrix = d['conn']
    
    irm_latent, irm_data = create_jobs_latent(conn_matrix['link'], 
                                              "BetaBernoulli")
    
    HPS = {'alpha' : 0.1,
           'beta' : 1.0}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, 
                open(meta_filename, 'w'))

@follows(dataset_create)
@files('dataset.pickle', ['jobs.ld.data', 'jobs.ld.latent', 'jobs.ld.meta'])
def create_latents_ld(infile, (data_filename, latent_filename, meta_filename)):
    d = pickle.load(open(infile, 'r'))
    conn_matrix = d['conn']
    
    irm_latent, irm_data = create_jobs_latent(conn_matrix, 
                                              "LogisticDistance")
    
    HPS = {'mu_hp' : 20.0,
           'lambda_hp' : 20.0,
           'p_min' : 0.001, 
           'p_max' : 0.9}

    irm_latent['relations']['R1']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile}, 
                open(meta_filename, 'w'))


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

        for domain_name in irm_data['domains']:
            d_N = len(latent['domains'][domain_name]['assignment'])
            if init['type'] == 'fixed':
                group_num = init['group_num']

                a = np.arange(d_N) % group_num
                a = np.random.permutation(a)

            elif init['type'] == 'crp':
                alpha = init['alpha']
                a = irm.util.crp_draw(d_N, alpha)
                a = np.random.permutation(a) 
            elif init['type'] == 'truth':
                a = latent['domains'][domain_name]['assignment']

            else:
                raise NotImplementedError("Unknown init type")

            if c > 0: # first one stays the same
                latent['domains'][domain_name]['assignment'] = a

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)

        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=2)


        pickle.dump(irm.irmio.get_latent(irm_model), open(out_f, 'w'))
            
def get_dataset(data_name):
    return glob.glob("%s.data" %  data_name)

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name,  i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]




@follows(create_latents_bb)
@follows(create_latents_ld)
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"

    create_init(latent_filename, data_filename, out_filenames, 
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
        t = np.array(d['times'])[::subsamp] - d['times'][0]
        ax_score.plot(t, s, alpha=0.7, c='k')

    f.tight_layout()

    f.savefig(plot_latent_filename)

@transform(get_results, suffix(".samples"), 
           [(".%d.latent.pdf" % d, )  for d in range(3)])
def plot_best_latent(exp_results, 
                     out_filenames):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    meta_filename = data_filename[:-4] + "meta"
    m = pickle.load(open(meta_filename, 'r'))
    meta_infile = m['infile']
    meta = pickle.load(open(meta_infile, 'r'))
    conn_matrix = meta['conn']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

    # get data
    
    for chain_pos, (latent_fname, ) in enumerate(out_filenames):
        best_chain_i = chains_sorted_order[chain_pos]
        best_chain = chains[best_chain_i]
        sample_latent = best_chain['state']
        jobs_assignment =  np.array(sample_latent['domains']['jobs']['assignment'])
        users_assignment = np.array(sample_latent['domains']['users']['assignment'])

        ji = np.argsort(jobs_assignment).flatten()
        ja = jobs_assignment[ji]
        j_pos = np.argwhere(np.diff(ja) != 0).flatten()

        ui = np.argsort(users_assignment).flatten()
        ua = users_assignment[ui]
        u_pos = np.argwhere(np.diff(ua) != 0).flatten()
        
        
        f = pylab.figure()
        ax = f.add_subplot(1, 1, 1)
        cm = conn_matrix['link']
        cm = cm[ui, :]
        cm = cm[:, ji]
        
        ax.imshow(cm, interpolation='nearest', cmap=pylab.cm.Greys)
        for i in u_pos:
            ax.axhline(i)

        for i in j_pos:
            ax.axvline(i)

        f.savefig(latent_fname)

if __name__ == "__main__":
    pipeline_run([create_data, dataset_create, dataset_debug, create_latents_bb, 
                  create_inits, run_exp, get_results, plot_scores_z, 
                  plot_best_latent])
