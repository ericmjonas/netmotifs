import numpy as np
import cPickle as pickle
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
import irm    

for dataset, model in [('xysregs', 'ldfl'), 
                       ('xysregs', 'ld'), 
                       ('xysregs', 'bb'), 
                       ('xysregs', 'ndfw'), 
                       ('decode', 'ldfl'), 
                       ('decode', 'ld'), 
                       ('lower', 'ldfl'), 
                       ('lower', 'ld'), 

]:

    raw_data = pickle.load(open("data.pickle", 'r'))
    wiredf = raw_data['wiredf']
    tfdf = raw_data['tfdf']

    x = pickle.load(open("typed.%s.region.pickle" % dataset))
    adj_mat = x['adj_mat']
    #typed_adjmat = pickle.load(open("typed.adjmat.pickle"))
    #pin_pairs = typed_adjmat['pin_pairs'] 

    # Should really load these from above file

    pin_pairs = [('gate', 'gate'), ('gate', 'c1'), ('gate', 'c2'), 
                 ('c1', 'c1'), ('c1', 'c2'), ('c2', 'c2')]


    RELATION_LIST = ["%s_%s" % a for a in pin_pairs]

    sample = pickle.load(open("data/mos6502.typed.%s.%s.data-fixed_100_200-anneal_slow_400.0.latent.pickle" % (dataset, model),  'r'))
    data = pickle.load(open("data/mos6502.typed.%s.%s.data" % (dataset, model),  'r'))

    print "main data loaded"

    df = x['subdf']
    print x['region']
    df['cluster'] = sample['domains']['d1']['assignment']
    for pin in ['gate', 'c1', 'c2']:
        df = df.join(wiredf['name'], on=[pin], rsuffix='.%s'% pin)
    print df.head()

    fid = open("sixrelation.%s.%s.output.html" % (dataset, model), 'w')

    for c_n, c in df.groupby('cluster'):
        fid.write(c.sort(['name.c1']).to_html())


    f = pylab.figure(figsize=(16, 16))
    ax = f.add_subplot(1, 1, 1)
    cpos = 0
    ax.scatter(df['x'], df['y'], c='k', s=10, edgecolor='none')
    for c_n, c in df.groupby('cluster'):
        if len(c) > 10 :
            ax.scatter(c['x'], c['y'], c = pylab.cm.jet(cpos), edgecolor='none', s=50)
            for row_name, row in c.iterrows():
                FS = 3
                ax.text(row['x'] + 12, row['y']+16, row_name, fontsize=FS)
                ax.text(row['x'] + 12, row['y'], "G:%s" % row['name.gate'], fontsize=FS)
                ax.text(row['x'] + 12, row['y']-16, "c1:%s" % row['name.c1'], fontsize=FS)
                ax.text(row['x'] + 12, row['y']-32, "c2:%s" % row['name.c2'], fontsize=FS)

            cpos += 0.3
    ax.set_xlim(df['x'].min()-50, df['x'].max()+50)
    ax.set_ylim(df['y'].min()-50, df['y'].max()+50)        
    f.tight_layout()
    f.savefig("sixrelation.%s.%s.transistorpos.pdf" % (dataset, model))

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
    f.savefig("sixrelation.%s.%s.latent.pdf"  % (dataset, model))

    pp = PdfPages("sixrelation.%s.%s.latents.pdf" % (dataset, model))
    all_r_f = pylab.figure(figsize=(16, 8))
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

        all_r_ax = all_r_f.add_subplot(2, 3, rel_i + 1)
        irm.plot.plot_t1t1_latent(all_r_ax, rel_data, assign)
        all_r_ax.set_title(rel_name)


        f.suptitle(rel_name)
        f.savefig(pp, format='pdf')

        f = pylab.figure(figsize=(16, 16))
        latent_rel_name = "R%d" % (rel_i + 1)
        print data.keys()
        irm.plot.plot_t1t1_params(f, rel_d, assign, 
                                  sample['relations'][latent_rel_name]['ss'], 
                                  sample['relations'][latent_rel_name]['hps'], 
                                  MAX_DIST = 2000, model=data['relations'][latent_rel_name]['model'])

        f.suptitle(rel_name)

        f.savefig(pp, format='pdf')

    pp.close()

    all_r_f.savefig("sixrelation.%s.%s.all.latents.pdf" % (dataset, model))


    circos_p = irm.plots.circos.CircosPlot(assign)
    irm.plots.circos.write(circos_p, "sixrelation.%s.%s.circos.png" % (dataset, model))

