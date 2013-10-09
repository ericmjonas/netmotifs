from ruffus import *
import pandas
from matplotlib import pylab
import cPickle as pickle
import preprocess

@files('apps.pickle', ['apps_per_user.pdf', 'users_per_job.pdf'])
def plot_apps_agg(infile, (apps_per_user_plot, users_per_job_plot)):
    apps_df = pickle.load(open(infile, 'r'))['apps']
    WINDOW_N = preprocess.WINDOW_N

    apps = apps_df[apps_df['WindowID'] == WINDOW_N]
    
    # plot jobs per applicant
    jc = apps['JobID'].value_counts()
    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(jc)
    f.savefig(apps_per_user_plot)

    vc_users = apps['UserID'].value_counts()

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(vc_users[:8000])
    ax.set_ylim(0, 50)
    ax.set_ylabel("# of applied jobs") 
    ax.set_xlabel("users")
    f.savefig(users_per_job_plot)

@files('zipcodes.pickle', 'zipcodes.pdf')
def plot_all_zipcodes(infile, outfile):
    f = pylab.figure(figsize=(20, 12))
    ax = f.add_subplot(1, 1, 1)
    zipcodes = pickle.load(open(infile, 'r'))
    zc = zipcodes['continental']
    ax.scatter(zc['longitude'], zc['latitude'], color='k', edgecolor='none', s=1)
    f.savefig(outfile)

if __name__ == "__main__":
    pipeline_run([plot_all_zipcodes, plot_apps_agg])

