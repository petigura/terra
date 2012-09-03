#!/usr/bin/env python
from argparse import ArgumentParser
import matplotlib
import tval

nbins = 20
prsr = ArgumentParser()

phelp = """
Parameters
pk file
"""
prsr.add_argument('pk'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()

if args.o:
    matplotlib.use('Agg')
from matplotlib.pylab import plt

pk   = args.pk
pk   = tval.Peak(pk)
pk.plot_diag()
if args.o is not None:
    plt.gcf().savefig(args.o)
else:
    plt.show()
    
