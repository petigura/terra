#!/usr/bin/env python
from argparse import ArgumentParser
prsr = ArgumentParser()

phelp = """
Parameters
pk file
"""
prsr.add_argument('pk'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()

import matplotlib
if args.o:
    matplotlib.use('Agg')

import kplot
from matplotlib.pylab import plt

pk   = args.pk
pk   = kplot.Peak(pk)
pk.plot_diag()
if args.o is not None:
    plt.gcf().savefig(args.o)
else:
    plt.show()
    
