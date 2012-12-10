from argparse import ArgumentParser
import sys
import sim
import pandas
import os
import terra
import matplotlib

class WritableObject:
    def __init__(self):
        self.content = []
    def write(self, string):
        self.content.append(string)

def injRecW(pardict,**kwargs):
    prefix = pardict['sid']+': '
    foo = WritableObject()                   # a writable object
    sys.stdout = foo                         # redirection
    out = {}
    try:
        out = terra.terra(pardict,**kwargs)
    except:
        import traceback    
        print  traceback.format_exc()
        out['id'] = pardict['id']

    sys.stdout = sys.__stdout__
    ostr = prefix+"".join(foo.content)
    ostr = ostr.replace('\n','\n'+prefix)
    print ostr
    return out

parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('parfile',type=str,help='file with the transit parameters')
parser.add_argument('outfile',type=str,help='output data here')
parser.add_argument('--DV',action='store_true')
args = parser.parse_args()
simPar = pandas.read_csv(args.parfile,index_col=0)
simPar['skic'] = ['%09d' %s for s in simPar['skic'] ]
if matplotlib.cbook.is_numlike(simPar['sid']):
    # Assume we're mean the skic
    simPar['sid'] = simPar['skic']

dL = []
for i in simPar.index:
    dL.append( injRecW( dict(simPar.ix[i]),startDV=args.DV) )

dL = pandas.DataFrame(dL)
simPar = pandas.merge(simPar,dL,how='left',on='id',suffixes=('_inp','_out'))
simPar.to_csv(args.outfile)

