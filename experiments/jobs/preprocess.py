from ruffus import *
import pandas
import os
import cPickle as pickle
import numpy as np


RAW_DATA_DIR = "../../../data/jobs"

WINDOW_N = 2

def zip_conv(x):
    if len(x) == 5:
        try:
            return float(x)
        except:
            return np.nan
    else:
        return np.nan

@files(os.path.join(RAW_DATA_DIR, "dl/users.tsv"), 
       "users.pickle")
def load_user_df(infile, outfile):
    
    users_df = pandas.io.parsers.read_csv(infile, sep='\t', index_col=0, 
                                          converters={"ZipCode" : zip_conv})
    pickle.dump({'users' : users_df}, 
                open(outfile, 'w'))

@files(os.path.join(RAW_DATA_DIR, "dl/apps.tsv"), 
       "apps.pickle")
def load_apps_df(infile, outfile):
    
    apps_df = pandas.io.parsers.read_csv(infile, sep='\t')
    pickle.dump({'apps' : apps_df}, 
                open(outfile, 'w'))


@files(os.path.join(RAW_DATA_DIR, "dl/splitjobs/jobs2.tsv"), 
       "jobs.%d.pickle" % WINDOW_N)
def load_jobs(infile, outfile):
    #jobs1 has an encoding error because everything remotely associated with tsv is a cl

    jobs_df = pandas.io.parsers.read_csv(infile, sep='\t', index_col=0, 
                                          converters={'Zip5' : zip_conv })
    pickle.dump({'jobs' : jobs_df}, 
                open(outfile, 'w'))

@files(os.path.join(RAW_DATA_DIR, "zip_codes_states.csv"), 
       "zipcodes.pickle")
def load_zipcodes(infile, outfile):
    zip_codes = pandas.io.parsers.read_csv(infile, index_col=0, dtype={'zip_code' : float})
    continental_long = [-130, -60]
    continental_lat = [20, 55]
    zip_codes_continental = zip_codes[(zip_codes['longitude'] > continental_long[0]) & (zip_codes['longitude'] < continental_long[1])]
    
    zip_codes_continental = zip_codes_continental[(zip_codes_continental['latitude'] > continental_lat[0]) & (zip_codes_continental['latitude'] < continental_lat[1])]
    pickle.dump({'all' : zip_codes, 
                 'continental' : zip_codes_continental}, 
                open(outfile, 'w'))

if __name__ == "__main__":
    pipeline_run([load_user_df, load_apps_df, load_jobs, 
                  load_zipcodes])
