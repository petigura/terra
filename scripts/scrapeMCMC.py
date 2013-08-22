import pandas as pd
import h5py
import tval
from argparse import ArgumentParser as ArgPar
import sys

parser = ArgPar(description='Grab the MCMC parameters, stick in csv')
parser.add_argument('infile',type=str,help='list of files to scrape')
parser.add_argument('outfile',type=str,help='file with parameters')
args  = parser.parse_args()

df = pd.read_table(args.infile,names=['file'])
dL = []
for i in df.index:
    file = 'grid/' + df.ix[i,'file']

    d = {}
    d['file'] = file
    try:
        with h5py.File(file) as h5:
            dtemp = tval.TM_getMCMCdict(h5)
            for k in dtemp.keys():
                d[k] = dtemp[k]
    except:
        print file,sys.exc_info()[1]

    dL += [d]
    if i % 100 == 0 : print i

df = pd.DataFrame(dL)
df.to_csv(args.outfile)
