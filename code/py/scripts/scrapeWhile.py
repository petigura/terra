#!/usr/bin/env python

import pandas as pd
import h5py
import sys
import os
from os.path import basename

import glob
import tval
from argparse import ArgumentParser

parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('start',type=int,help='start')
parser.add_argument('stop',type=int,help='stop')
args  = parser.parse_args()

pp = pd.read_csv('pp.csv',index_col=0)
pp = pp.iloc[range(args.start,args.stop)]

fL = glob.glob('../www/K2/TPS/C0/*.h5')
df = pd.DataFrame(fL,columns=['file'])

df['outbase'] = df.file.apply(basename)
pp['outbase'] = pp.outfile.apply(basename)

pp = pd.merge(df[['outbase']],pp)

dL = []
for i in pp.index:
    outf = pp.ix[i,'outfile']
    d = {}
    d['outfile'] = outf
    try:
        with h5py.File(outf,mode='r') as h5:
            d['num_trans']  = h5.attrs['num_trans']
            ktop = 'P autor s2ncut s2n medSNR t0 skic grass'.split()
            for k in ktop:
                d[k]  = h5.attrs[k]

            dtemp = tval.TM_getMCMCdict(h5)
            for k in dtemp.keys():
                d[k] = dtemp[k]

    except:
        print outf,sys.exc_info()[1]
    if i%10==0:
        print i
    dL.append(d)

df = pd.DataFrame(dL)
df.to_csv('scrape.out.csv.%06d-%06d' % (args.start,args.stop))
