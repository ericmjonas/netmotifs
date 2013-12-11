import numpy as np
import cPickle as pickle
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
import irm    
import pandas

CIRCOS_DIST_THRESHOLDS = [20, 50, 100, 200, 500]

for dataset, model in [('xysregs', 'ldfl'), 
                       #('xysregs', 'ld'), 
                       #('xysregs', 'ndfw'), 
                       #('xysregs', 'bb'), 
                       #('decode', 'ldfl'), 
                       #('decode', 'ld'), 
                       #('lower', 'ldfl'), 
                       #('lower', 'ld')
]:

    raw_data = pickle.load(open("data.pickle", 'r'))
    wiredf = raw_data['wiredf']
    tfdf = raw_data['tfdf']

    null_names = wiredf['name'].isnull()
    wiredf['name'][null_names] = ["n%d" % d for d in range(np.sum(null_names))]
    # translations
    wire_translations = {"dasb0": "sb0", 
                         "dasb4" : "sb4"}
    for w_old, w_new in wire_translations.iteritems():

        wiredf['name'][wiredf['name'] == w_old] = w_new


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




    df = x['subdf']
    print x['region']
    assign = np.array(sample['domains']['d1']['assignment'])

    class_ids = np.unique(assign)
    CLASS_N = len(class_ids)


    custom_color_map = {}
    custom_color_map_raw = {}
    for c_i, c_v in enumerate(class_ids):
        color = pylab.cm.Set1(float(c_i) / (CLASS_N+1))
        c = np.array(color)[:3]*255
        custom_color_map['ccolor%d' % c_v]  = c.astype(int)
        custom_color_map_raw[c_v] = color

    df['cluster'] = assign
    for pin in ['gate', 'c1', 'c2']:
        df = df.join(wiredf['name'], on=[pin], rsuffix='.%s'% pin)

    fname = "sixrelation_circos.%s.%s.xlsx" % (dataset, model)
    writer= pandas.ExcelWriter(fname)
    for col in ['bb', 'c1', 'c2', 'gate', 'name', 'on']:
        del df[col]
    for c_n, c in df.groupby('cluster'):
        c_df = c.sort(['name.gate', 'name.c1'])
        del c_df['cluster']
        c_df.to_excel(writer, sheet_name='cluster%d' % c_n)
    writer.save()

    # now formatting stuff 
    from openpyxl import Workbook, load_workbook
    from openpyxl.style import Fill, Color
    
    wb = load_workbook(filename = fname)
    for c_n, c in df.groupby('cluster'):
        sheet_name = 'cluster%d' % c_n
        sheet = wb.get_sheet_by_name(name = sheet_name)
        sheet.column_dimensions['B'].width = 10
        sheet.column_dimensions['C'].width = 10
        sheet.column_dimensions['D'].width = 18
        sheet.column_dimensions['E'].width = 18
        sheet.column_dimensions['F'].width = 18
        hc  =  np.array(custom_color_map_raw[c_n]) * 255
        print custom_color_map_raw[c_n], hc
        hex_str = "%02x%02x%02x%02x" % (hc[3], hc[0], hc[1], hc[2])
        for row in sheet.range("A1:A%d" % (len(c) + 1)):
            for cell in row:
                cell.style.fill.fill_type = Fill.FILL_SOLID
                cell.style.fill.start_color.index = hex_str

    wb.save(fname)


