"""
Simple program for plotting where a target is on the Kepler FOV
"""

import stellar
from optparse import OptionParser
import sys
from argparse import ArgumentParser
from matplotlib.pylab import *

def main():
    psr = ArgumentParser(description='Compute photometry from K2 data')
    psr.add_argument('campaign',type=str,help='Campaign')
    psr.add_argument('epic',type=int,help='EPIC ID')
    args = psr.parse_args()

    targets = stellar.read_cat(k2_camp=args.campaign)
    plot(targets.ra,targets.dec,',')

    epic = args.epic
    star = targets.ix[epic]
    plot(star.ra,star.dec,'o',color='Tomato')
    setp(gca(),xlabel='RA (deg)',ylabel='Dec (deg)',
         title='%i on Kepler FOV' % epic)

    show()

    return

if __name__ == "__main__":
    sys.exit(main())
