from argparse import ArgumentParser
import h5plus
import photometry
import stellar
import h5py
import numpy as np
import prepro

psr = ArgumentParser(description='Compute photometry from K2 data')
psr.add_argument('--np',type=int,default=1,help='Number of processors to use')
args = psr.parse_args()

df = stellar.read_cat()
nlc = len(df)

f = 'C0.h5'
ts,cube = photometry.read_pix(f,df.iloc[0]['epic'])
lc = photometry.circular_photometry(ts,cube)
lc0 = prepro.rdt(lc)

h5file = 'Ceng.h5'

with h5plus.File(h5file) as h5:
    h5.create_dataset('dt',dtype=lc0.dtype,shape=(len(df),lc0.size) )
    h5['epic'] = np.array(df.epic)



numpro  = args.np
ids = np.arange(nlc)
if numpro==1:
    for i in ids:
        lcL = get_lc(i)
        with h5py.File(h5file) as h5:
            h5['dt'][i,:] = lcL[0]
else:
    def circ_phot(arg):
        ts = arg[0]
        cube = arg[1]
        return photometry.circular_photometry(ts,cube)

    # Using Pool.map to handle parallelism. 
    from multiprocessing import Pool
    pool = Pool(numpro)
    idsL = np.array_split(ids,nlc / numpro)
    for iL in idsL:
        tsL,cubeL = photometry.read_pix(f,df.iloc[iL]['epic'])

        lcL = pool.map(circ_phot,zip(tsL,cubeL))
        # For some reason prepro.rdt screws up multiprocessing module.
        lcL = map(prepro.rdt,lcL)
        lcL = np.vstack(lcL)

        with h5py.File(h5file) as h5:
            h5['dt'][iL,:] = lcL


        print df.iloc[iL]['epic']

        
    pool.close()
    pool.join()
    



