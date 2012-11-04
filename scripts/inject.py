from argparse import ArgumentParser
import sys
import sim
import pandas

class WritableObject:
    def __init__(self):
        self.content = []
    def write(self, string):
        self.content.append(string)

def injRecW(pardict):
    prefix = pardict['bname']+': '
    foo = WritableObject()                   # a writable object
    sys.stdout = foo                         # redirection
    out = {}
    try:
        out = sim.injRec(pardict)
    except:
        import traceback    
        print  traceback.format_exc()
        out['id'] = pardict['id']

    sys.stdout = sys.__stdout__
    ostr = prefix+"".join(foo.content)
    ostr = ostr.replace('\n','\n'+prefix)
    print ostr
    return out

parser = ArgumentParser(description='Inject in raw LC; compute DV parameters')
parser.add_argument('parfile',type=str,help='file with the transit parameters')
parser.add_argument('outfile',type=str,help='output data here')
args = parser.parse_args()
simPar = pandas.read_csv(args.parfile,index_col=0)
simPar['id'] = simPar.index
simPar['pngfile'] = simPar['bname']+'.png'
#dL = map(injRecW,[simPar.ix[i] for i in simPar.index ] )
dL = map(sim.injRec,[simPar.ix[i] for i in simPar.index ] )
dL = pandas.DataFrame(dL)
simPar = pandas.merge(simPar,dL,how='left',on='id',suffixes=('_inp','_out'))
simPar.to_csv(args.outfile)
