from ruffus import *
import pandas
from matplotlib import pylab
import cPickle as pickle
import preprocess
import numpy as np


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
    ax = f.add_subplot(1, 1, 1)
    
    ax.hist(data['conn']['distance'].flatten(), bins=20)
    f.savefig(outfile)

    
if __name__ == "__main__":
    pipeline_run([create_data, dataset_create, dataset_debug])
