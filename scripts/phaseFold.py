#!/usr/common/usg/python/2.7.1-20110310/bin/python
from argparse import ArgumentParser
import pandas
import h5py
import tval
from matplotlib import mlab
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import kplot
import matplotlib.pylab as plt
import morton


parser = ArgumentParser(description='Phase Fold Lightcurves')
parfilehelp="""
parfile is a csv with the following names
- KOI : the index
- P,t0,df,tduy
- lcfile,pkfile
"""

parser.add_argument('cat',type=str,help='')
parser.add_argument('koi',type=str,help='')
args = parser.parse_args()

cat = args.cat
if cat=='CB':
    row = morton.getParCB(args.koi)
else:
    row = morton.getParJR(args.koi)

morton.phaseFoldKOI(row)
print "created: %(pkname)s" % row

with h5py.File(row['pkname']) as pk:
    kplot.morton(pk)
    ax = plt.gca()

    ds = pandas.Series(row)
    txt = ds.drop(['lcname','pkname','name']).to_string()
    at = AnchoredText(txt,prop=dict(size='medium'),frameon=True,loc=4)
    ax.add_artist(at)

    plt.show()
    plt.gcf().savefig(row['pkname'].replace('.h5','.png'))
