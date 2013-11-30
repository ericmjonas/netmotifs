import numpy as np
import cPickle as pickle
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
import irm    

CIRCOS_DIST_THRESHOLDS = [20, 50, 100, 200, 500]

for dataset, model in [('xysregs', 'ldfl'), 
                       ('xysregs', 'ld'), 
                       ('xysregs', 'ndfw'), 
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

            def conv(s):
                out = []
                for n in s:
                    a = 'A'
                    if n == 'cclk':
                        a = 'C'
                    elif n == 'vss':
                        a = 'O'
                    # elif n in ['x%d' % d for d in range(8)]:
                    #     a = 'I'
                    # elif n in ['y%d' % d for d in range(8)]:
                    #     a = 'I'
                    # elif n in ['s%d' % d for d in range(8)]:
                    #     a = 'I'
                    elif type(n) == str and (('not' in n) or ('~' in n)):
                        a = 'I'
                    out.append(a)



                return out

            def clean(x):
                out = []
                for y in x:
                    if type(y) == float:
                        out.append('_')
                    elif '#' in y:
                        out.append("n%s" % y[1:] )
                    else:
                        out.append(y)
                return out

            # # add pin glyphs for 'vss' and 'clk'
            # circos_p.add_plot('text', {'r0' : '1.05r', 
            #                            'r1' : '1.10r', 
            #                            'label_size' : '20p', 
            #                            'label_font' : 'glyph', 
            #                            'label_rotate' : 'yes', 
            #                        }, 
            #                   conv(df['name.c2']))

            # circos_p.add_plot('text', {'r0' : '1.10r', 
            #                            'r1' : '1.15r', 
            #                            'label_size' : '20p', 
            #                            'label_font' : 'glyph', 
            #                            'label_rotate' : 'yes', 
            #                        }, 
            #                   conv(df['name.c1']))

            # circos_p.add_plot('text', {'r0' : '1.15r', 
            #                            'r1' : '1.20r', 
            #                            'label_size' : '20p', 
            #                            'label_font' : 'glyph', 
            #                            'label_rotate' : 'yes', 
            #                        }, 
            #                   conv(df['name.gate']))

            circos_p.add_plot('text', {'r0' : '1.05r', 
                                       'r1' : '1.15r', 
                                       'label_font' : 'condensed', 
                                       'label_size' : '20p', 
                                       'label_rotate' : 'yes', 
                                   }, 
                              clean(df['name.c1']))


            circos_p.add_plot('text', {'r0' : '1.15r', 
                                       'r1' : '1.25r', 
                                       'label_font' : 'condensed', 
                                       'label_size' : '20p', 
                                       'label_rotate' : 'yes', 
                                   }, 
                              clean(df['name.c2']))


            circos_p.add_plot('text', {'r0' : '1.25r', 
                                       'r1' : '1.35r', 
                                       'label_font' : 'condensed', 
                                       'label_size' : '20p', 
                                       'label_rotate' : 'yes', 
                                   }, 
                              clean(df['name.gate']))

            irm.plots.circos.write(circos_p, "sixrelation.%s.%s.circos.%d.%s.pdf" % (dataset, model, fi, relation))


            # now the tiny plots
        
            circos_p = irm.plots.circos.CircosPlot(assign, ideogram_radius="0.4r", 
                                                   ideogram_thickness="60p")
            circos_p.add_class_ribbons(ribbons, color)
            irm.plots.circos.write(circos_p, "sixrelation.%s.%s.circos.%d.%s.tiny.pdf" % (dataset, model, fi, relation))
