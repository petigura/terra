import matplotlib
matplotlib.use('Agg')

from argparse import ArgumentParser
import h5plus
import photometry
import stellar
import h5py
import numpy as np
import prepro

from config import path_pix_fits,path_phot
from matplotlib.pylab import *

psr = ArgumentParser(description='Compute photometry from K2 data')
psr.add_argument('--np',type=int,default=1,help='Number of processors to use')
args = psr.parse_args()

df = stellar.read_cat()
nlc = len(df)

def get_lc(i):
    epic = df.iloc[i]['epic']
    f = '%s/kplr%09d-2014044044430_lpd-targ.fits' % (path_pix_fits,epic)
    ts,cube,aper,head0,head1,head2 = photometry.read_k2_fits(f)
    lc = photometry.circular_photometry(ts,cube,aper,plot_diag=True)
    if path_phot.find('Ceng2C0')!=-1:
        lc = photometry.Ceng2C0(lc)
    return lc


lc0 = get_lc(0)
with h5plus.File(path_phot) as h5:
    h5.create_dataset('dt',dtype=lc0.dtype,shape=(len(df),lc0.size) )
    h5['epic'] = np.array(df.epic)
    

numpro  = args.np
ids = np.arange(nlc)
if numpro==1:
    for i in ids:
        epic = df.iloc[i]['epic']
        lc = get_lc(i)

#        gcf().savefig('Ceng/plots/%09d.png' % epic)

        with h5py.File(path_phot) as h5:
            h5['dt'][i,:] = lc

        if (i%10)==0:
            print i
        
