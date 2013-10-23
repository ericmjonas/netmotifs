from ruffus import *
import pandas
import os
import cPickle as pickle
import numpy as np
import shapefile


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


@merge(os.path.join(RAW_DATA_DIR, "dl/splitjobs/jobs*.tsv"), 
       "jobs.pickle")
def load_jobs(infiles, outfile):

    jobs = []
    USECOLS = [0, 1, 2,  4, 5, 6, 7, 8, 9, 10] # drop descr
    for infile in infiles:
        jobs.append(pandas.io.parsers.read_csv(infile, sep='\t', index_col=0, 
                                               converters={'Zip5' : zip_conv }, 
                                               usecols= USECOLS,
                                               error_bad_lines=False))
    jobs_df = pandas.concat(jobs)
    pickle.dump({'jobs' : jobs_df}, 
                open(outfile, 'w'))

@files([load_user_df, load_apps_df, load_jobs], 'appszips.pickle')
def clean_merge_apps((users_filename, apps_filename, jobs_filename), 
                     out_filename):
    """
    Clean up the apps, merge in zipcodes, 
    """

    apps_df = pickle.load(open(apps_filename, 'r'))['apps']
    users_df = pickle.load(open(users_filename, 'r'))['users']
    jobs_df = pickle.load(open(jobs_filename, 'r'))['jobs']
    jobs_zips = jobs_df['Zip5']
    users_zips = users_df['ZipCode']
    apps_df = apps_df.join(users_zips, on='UserID')
    apps_df = apps_df.join(jobs_zips, on='JobID')
    
    apps_df = apps_df.dropna(subset=['UserID', 'JobID'])

    
    pickle.dump({'apps' : apps_df}, 
                open(out_filename, 'w'))

                 
    
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

@files(os.path.join(RAW_DATA_DIR, "zips"), "zipcodeshapes.pickle")
def zipcodeshapes(indir, outfile):
    sf = shapefile.Reader(indir + "/zip_codes_for_the_usa")

    shapes = sf.shapes()

    records = sf.records()

    data = []
    for shape, record in zip(shapes, records):
        zip_str = record[1]
        data.append({'zipcode' : zip_str, 
                     'points' : shape.points})

    df = pandas.DataFrame(data)
    df = df.set_index(df['zipcode'])
    pickle.dump(df, open(outfile, 'w'))


if __name__ == "__main__":
    pipeline_run([load_user_df, load_apps_df, load_jobs, 
                  clean_merge_apps, 
                  load_zipcodes, zipcodeshapes])
