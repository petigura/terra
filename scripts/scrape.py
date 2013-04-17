import pandas,h5py
import os,sys

pp = pandas.read_csv('pp.csv',index_col=0)
dL = []
for i in pp.outfile:
    d = {}
    d['outfile'] = i
    try:
        with h5py.File(i,mode='r') as h5:
            l = h5.keys()
            o = ''
            if l.index('cal')!=0:
                o = 'cal' 
            if l.index('it0')!=0:
                o = 'it0'
            d['last'] = o
            ktop = 'P,autor,s2ncut,s2n,medSNR,t0,skic'.split(',')
            for k in ktop:
                d[k]  = h5.attrs[k]
            d['p0']   = h5['fit'].attrs['pL0'][0]
            d['tau0'] = h5['fit'].attrs['pL0'][1]
            d['b0']   = h5['fit'].attrs['pL0'][2]
    except:
        print i,sys.exc_info()[1]
    dL.append(d)
    
df = pandas.DataFrame(dL)
df.to_csv(os.getcwd()+'.out.csv')
