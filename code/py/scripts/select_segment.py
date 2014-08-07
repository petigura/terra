path = '/project/projectdirs/m1669/Kepler/lcQ14/'

fL = [
'005817791.h5',
'007461298.h5',
'008804435.h5',
'010323549.h5',
'012645262.h5']
from matplotlib.pylab import *
import h5py
import prepro

ys = 0
dy = 0.03

color=['Tomato','RoyalBlue']
clf()

def getseg(lc):
    seglabel = np.zeros(lc.size) - 1
    t = lc['t']
    tm = ma.masked_array(t)

    for i in range( len(prepro.cutList)-1 ):
        rng  = prepro.cutList[i]
        rng1 = prepro.cutList[i+1]

        tm = ma.masked_inside(tm,rng['start'],rng['stop'])

        b = ( tm > rng['stop'] ) & ( tm < rng1['start'] ) 
        seglabel[b] = i
    return seglabel


def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 10
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * 0.03 * sqdist)

def bin(lc):
    fm = ma.masked_invalid( lc['f'] )
    nbin = 8
    rem  = remainder(lc.size,nbin)
    if rem > 0:
        npad = nbin - rem
        pad  = ma.masked_array(np.zeros(npad),True)
        fm = ma.hstack([fm,pad])

    y   = fm.reshape(-1,nbin).mean(axis=1)
    x   = lc['t'][::nbin]
    b = ~y.mask
    return x[b],y.data[b]

def GPdt(xi,x,y):
    X  = x[:,np.newaxis]
    Xi = xi[:,np.newaxis]

    K = kernel(X,X)
    s = 0.005    # noise variance.
    N = len(X)   # number of training points.
    L = np.linalg.cholesky(K + s*np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xi) )
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    return mu

for j in range(len(fL)):
    f = fL[j]
    with h5py.File(path+f) as h5:
        for i in h5['raw'].items():
#        for i in [h5['/raw']['Q14']] : 
            lc = i[1][:]
            t = lc['t']
            
            seglabel = getseg(lc)

            lw = 3
            f = lc['f']
            plot(t,f-ys,color='k',alpha=0.4,lw=lw)

            fm = ma.masked_array(f)
            fm.mask = (seglabel == -1) | (seglabel %2 ==1)
            plot(t,fm-ys,color='Tomato',lw=lw)            

            fm = ma.masked_array(f)
            fm.mask = (seglabel == -1) | (seglabel %2 ==0)
            plot(t,fm-ys,color='RoyalBlue',lw=lw)            
            axvline(t[0],alpha=0.1,color='c',lw=lw)

            sL = unique(seglabel)
            sL = sL[sL >=0]
            for s in sL:
                lc2 = lc[seglabel==s]
                x,y = bin(lc2)

                yi  = GPdt(lc2['t'],x,y)
                plot(lc2['t'],yi-ys,'c',lw=2)
    ys +=dy

draw()

