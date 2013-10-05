import pandas as pd
import h5py
import tval
from argparse import ArgumentParser as ArgPar
import sys
import VShape
from matplotlib.pylab import *
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


parser = ArgPar(description='Grab the MCMC parameters, stick in csv')
parser.add_argument('infile',type=str,help='list of files to scrape')
args  = parser.parse_args()

h5 = h5py.File(args.infile)
trans = VShape.TM_read_h5(h5)
trans.register()
trans.pdict = trans.fit()[0]


model = trans.trap(trans.pdict,trans.t)
plot(trans.t,trans.f,'.')
plot(trans.t,model)
plot(trans.t,trans.f-model)
xlabel('t - t0 (days)') 
ylabel('flux') 

s = """\
df   %(df).2e
fdur %(fdur).3f
wdur %(wdur).3f
dt   %(dt).3f""" % trans.pdict

print args.infile.split('/')[-1][:9] + \
    " %(df).2e %(fdur).3f %(wdur).3f %(dt).3f" % trans.pdict
at = AnchoredText(s,4,prop=dict(family='monospace'));gca().add_artist(at)
gcf().savefig(args.infile.replace('h5','vs.png'))
