import numpy as np
import cPickle as pickle
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
import irm    

CIRCOS_DIST_THRESHOLDS = [20, 50, 100, 200, 500]

for dataset, model in [('xysregs', 'ldfl'), 
                       ('xysregs', 'ld'), 
                       #('xysregs', 'bb'), 
                       #('decode', 'ldfl'), 
                       #('decode', 'ld'), 
                       #('lower', 'ldfl'), 
                       #('lower', 'ld')
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
        fid.write(c.sort('name.gate').to_html())

    model_name = data['relations']['R1']['model']


    assign = np.array(sample['domains']['d1']['assignment'])
    if "istance" not in model_name:
        continue
    for fi in range(len(CIRCOS_DIST_THRESHOLDS)):

        for relation, color in [('R1', 'black_a5'), 
                                ('R2', 'black_a5'), 
                                ('R3', 'black_a5'),
                                ('R4', 'black_a5'), 
                                ('R5', 'black_a5'), 
                                ('R6', 'black_a5'), 
                            ]:
            circos_p = irm.plots.circos.CircosPlot(assign)


            a = irm.util.canonicalize_assignment(assign)
            v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
                                               sample['relations'][relation]['ss'], 
                                               sample['relations'][relation]['hps'], 
                                               model_name)
            thold = 0.3
            ribbons = []
            links = []
            pairs_plotted = set()
            
            for (src, dest) in v.keys():
                p1 = v[(src, dest)]
                p2 = v[(dest, src)]
                p = max(p1, p2)
                if (src, dest) in pairs_plotted or (dest, src) in pairs_plotted:
                    pass
                else:
                    
                    if p > thold :
                        pix = int(20*p)
                        print src, dest, p, pix

                        ribbons.append((src, dest, pix))

                pairs_plotted.add((src, dest))

            circos_p.add_class_ribbons(ribbons, color)


            irm.plots.circos.write(circos_p, "sixrelation.%s.%s.circos.%d.%s.svg" % (dataset, model, fi, relation))

