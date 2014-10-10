#!/usr/bin/env python
import pandas as pd
import h5py
import sys
import sqlite3
import glob
import tval
import os
from argparse import ArgumentParser
p = ArgumentParser(description='Scrape attributes from grid.h5 files')
p.add_argument('pardb',type=str,help='Path to pars.sqlite database')
p.add_argument(
    'outputdir',type=str,
    help='Output directory containing <star>/<star>.grid.h5')
p.add_argument('--start',type=int,help='start')
p.add_argument('--stop',type=int,help='stop')
args  = p.parse_args()

outputdir = args.outputdir
outputdir = os.path.abspath(outputdir)
pardir = os.path.abspath(os.path.join(outputdir, os.pardir))

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
fL = glob.glob("%s/*/*.grid.h5" % outputdir)
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
        d = tval.scrape(x['file'],verbose=False)                
    except:
        print outf,sys.exc_info()[1]
        
    if (x['counter'] % 10)==0:
        print x['counter']

    df+=[d]
        
df = pd.DataFrame(df)

csvfile = 'scrape.out.csv.%06d-%06d' % (ilocs[0],ilocs[-1])
csvfile = os.path.join(pardir,csvfile)
df.to_csv(csvfile)

sqlitefile = csvfile.replace('csv','sqlite')
con = sqlite3.connect(sqlitefile)
df.to_sql('scrape',con,if_exists='replace')
