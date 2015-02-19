import pandas,h5py
import os,sys
pp = pandas.read_csv('pp.csv',index_col=0)
dL = []
for i in pp.index:
    outf = pp.ix[i]['outfile']
    pd = dict(pp.ix[i])
    d = {}
    d['outfile'] = outf
    try:
        with h5py.File(outf,mode='r') as h5:
            d['num_trans']  = h5.attrs['num_trans']
            ktop = 'P,autor,s2ncut,s2n,medSNR,t0,skic'.split(',')
            for k in ktop:
                d[k]  = h5.attrs[k]
            d['p0']        = h5['fit'].attrs['pL0'][0]
            d['tau0']      = h5['fit'].attrs['pL0'][1]
            d['b0']        = h5['fit'].attrs['pL0'][2]
    except:
        print outf,sys.exc_info()[1]
    if i%10==0:
        print i
    dL.append(d)
df = pandas.DataFrame(dL)
df.to_csv(os.getcwd()+'.out.csv')

