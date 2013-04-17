import matplotlib
from argparse import ArgumentParser
prsr = ArgumentParser()

prsr.add_argument('inp'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
args  = prsr.parse_args()

if args.o:
    matplotlib.use('Agg')

import h5py
import kplot
from matplotlib.pylab import plt

with h5py.File(args.inp) as h5:
    kplot.plot_diag(h5)
    if args.o is not None:
        plt.gcf().savefig(args.o)
    else:
        plt.show()
    
