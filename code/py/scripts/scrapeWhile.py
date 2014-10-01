#!/usr/bin/env python
import pandas as pd
import h5py
import sys
import sqlite3
import glob
import tval
from argparse import ArgumentParser

p = ArgumentParser(description='Scrape attributes from grid.h5 files')
p.add_argument('pardb',type=str,help='Path to pars.sqlite database')
p.add_argument(
    'outputdir',type=str,
    help='Output directory containing <star>/<star>.grid.h5')
p.add_argument('--start',type=int,help='start')
p.add_argument('--stop',type=int,help='stop')
args  = p.parse_args()


# Figure out which files we're supposed to read in
print "Reading in parameter file %s" % args.pardb
con = sqlite3.connect(args.pardb)
pp = pd.read_sql('select * from pp',con,index_col='id')
if args.start != None:
    ilocs = range(args.start,args.stop)
else:
    ilocs = range(len(pp))
pp = pp.iloc[ilocs]

# Grab the paths to the *.grid.h5 files
print "Searching for extant *.grid.h5 files"
fL = glob.glob("%s/*/*.grid.h5" % args.outputdir)
df = pd.DataFrame(fL,columns=['file'])
df['id'] = df.file.apply(lambda x : x.split('/')[-1].split('.')[0])

# Scrape only extant files
pp = pd.merge(pp,df,how='left',left_index=True,right_on='id')
n_files_extant = pp.file.notnull().sum()
print "%i extant files out of %i requested"  % (n_files_extant,len(pp) )

pp['counter'] = range(len(pp))
pp.index=pp.id

df = []
for i in pp.index:
    x = pp.ix[i]
    outf = x['file']
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

        
    if (x['counter'] % 10)==0:
        print x['counter']

    df+=[d]
        
df = pd.DataFrame(df)
df.to_csv('scrape.out.csv.%06d-%06d' % (ilocs[0],ilocs[-1]))
