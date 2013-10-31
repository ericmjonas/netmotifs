import numpy as np
import cPickle as pickle
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
import irm    

raw_data = pickle.load(open("data.pickle", 'r'))
wiredf = raw_data['wiredf']
tfdf = raw_data['tfdf']

x = pickle.load(open("typed.xysregs.region.pickle"))
adj_mat = x['adj_mat']
#typed_adjmat = pickle.load(open("typed.adjmat.pickle"))
#pin_pairs = typed_adjmat['pin_pairs'] 

# Should really load these from above file

pin_pairs = [('gate', 'gate'), ('gate', 'c1'), ('gate', 'c2'), 
             ('c1', 'c1'), ('c1', 'c2'), ('c2', 'c2')]


RELATION_LIST = ["%s_%s" % a for a in pin_pairs]

sample = pickle.load(open("data/mos6502.typed.xysregs.ld.data-fixed_100_200-anneal_slow_400.0.latent.pickle", 'r'))
print "main data loaded"

df = x['subdf']
print x['region']
df['cluster'] = sample['domains']['d1']['assignment']
for pin in ['gate', 'c1', 'c2']:
    df = df.join(wiredf['name'], on=[pin], rsuffix='.%s'% pin)
print df.head()

fid = open("output.html", 'w')

for c_n, c in df.groupby('cluster'):
    fid.write(c.sort('name.gate').to_html())


f = pylab.figure(figsize=(16, 16))
ax = f.add_subplot(1, 1, 1)
cpos = 0
ax.scatter(df['x'], df['y'], c='k', s=10, edgecolor='none')
for c_n, c in df.groupby('cluster'):
    if len(c) > 10 :
        ax.scatter(c['x'], c['y'], c = pylab.cm.jet(cpos), edgecolor='none', s=50)
        for row_name, row in c.iterrows():
            ax.text(row['x'] + 10, row['y']+10, row_name, fontsize=6)

        cpos += 0.3
        
f.savefig("sixrelation.transistorpos.pdf")

f = pylab.figure(figsize=(20, 20))
ax = f.add_subplot(1, 1, 1)

assign = np.array(sample['domains']['d1']['assignment'])
ai = np.argsort(assign).flatten()
agg_cnt = np.zeros((len(assign), len(assign)))
for ni, n in enumerate(RELATION_LIST):
    agg_cnt += adj_mat[n]*(2**(ni+1))
agg_cnt = agg_cnt[ai]
agg_cnt = agg_cnt[:, ai]
ax.imshow(agg_cnt, interpolation='nearest', cmap=pylab.cm.gist_ncar)
for a in np.argwhere(np.diff(assign[ai]) > 0).flatten():
    ax.axhline(a)
    ax.axvline(a)
f.savefig("sixrelation.latent.pdf")

pp = PdfPages("sixrelation.latents.pdf")

# now plot each block independently
for rel_i, rel_name in enumerate(RELATION_LIST):
    print "plotting for", rel_name
    rel_data = adj_mat[rel_name]
    # create empty data 
    rel_d = np.zeros(rel_data.shape, 
                     dtype = [('link', np.uint8), 
                              ('distance', np.float32)])
    rel_d['link'] = rel_data
    rel_d['distance'] = adj_mat['distance']
    f = pylab.figure(figsize=(16, 16))
    ax = f.add_subplot(1, 1, 1)
    irm.plot.plot_t1t1_latent(ax, rel_data, assign)
    f.suptitle(rel_name)
    f.savefig(pp, format='pdf')
    
    print sample['relations'].keys()
    f = pylab.figure(figsize=(16, 16))
    latent_rel_name = "R%d" % (rel_i + 1)
    irm.plot.plot_t1t1_params(f, rel_d, assign, 
                              sample['relations'][latent_rel_name]['ss'], 
                              sample['relations'][latent_rel_name]['hps'], 
                         MAX_DIST = 1000)

    f.suptitle(rel_name)
    
    f.savefig(pp, format='pdf')
    
pp.close()
    
