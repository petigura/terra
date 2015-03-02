#cython: boundscheck=False, wraparound=False

"""
Cython functions for folding
"""
import cython
cimport numpy as np

import numpy as np
from numpy import ma
from libc.math cimport exp, sqrt, pow, log, erf

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

IDTYPE = np.int64
ctypedef np.int64_t IDTYPE_t



cpdef fold_col(np.ndarray[np.float64_t, ndim=1] data,
              np.ndarray[np.int64_t, ndim=1] mask, 
              np.ndarray[np.int64_t, ndim=1] col):
    """
    Fold Columns

    Assign each point of time series to a column (bin) compute the
    following aggregating statristics for each column.

    Parameters
    ----------
    data : (float) data array
    mask : (int) mask for data array. 1 = masked out.
    col : (int) column corresponding to phase bin of measurement.

    Return
    ------
    ccol : total number of non-masked elements
    scol : sum of elements
    sscol : sum of squares of elements
    """

    cdef int icad, icol,ncad, ncol
    ncad = data.shape[0]
    ncol = np.max(col)+1
    
    # Define column arrays
    cdef np.ndarray[np.int64_t] ccol = np.zeros(ncol,dtype=int)
    cdef np.ndarray[np.float64_t] scol = np.zeros(ncol)
    cdef np.ndarray[np.float64_t] sscol = np.zeros(ncol)

    # Loop over cadences
    for icad in range(ncad):
        if (mask[icad]==0):
            icol = col[icad]

            # Increment counters
            ccol[icol]+=1 
            scol[icol]+=data[icad] 
            sscol[icol]+=data[icad]**2

    return ccol,scol,sscol

def wrap_icad(icad,Pcad):
    """
    rows and column identfication to each one of the
    measurements in df

    Parameters
    ----------
    icad : Measurement number starting with 0
    Pcad : Period to fold on
    """

    row = np.floor( icad / Pcad ).astype(int)
    col = np.floor(np.mod(icad,Pcad)).astype(int)
    return row,col


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef bls(np.ndarray[np.int64_t, ndim=1] ccol,
          np.ndarray[np.float64_t, ndim=1] scol, 
          np.ndarray[np.float64_t, ndim=1] sscol,
          int ncol,int twd1,int twd2):
    
    """
    Loop over different starting epocs
    """
    cdef int twdmax,colmax,col0,col1,twd,c
    cdef float s,ss,s2nmax,mean,std,noise,cfloat

    s2nmax = -1.
    twdmax = -1
    colmax = -1
    cdef noisemax = -1
    cdef meanmax = -1 
    for col0 in range(ncol):
        c = 0
        twd = 1
        s = 0.
        ss = 0.

        col1 = col0
        for i in range(twd2):
            c+=ccol[col1]
            s+=scol[col1]
            ss+=sscol[col1]
            cfloat = <float>c
            
            if (twd >= twd1) and (c > twd):
                # Compute mean transit depth and scatter
                mean = s / cfloat
                std = sqrt( (cfloat*ss-s**2) / (cfloat * (cfloat - 1)))
                noise = std / sqrt(cfloat)
                s2n = -1.0 * mean / noise

                if (s2n > s2nmax):
                    s2nmax = s2n
                    twdmax = twd
                    colmax = col0
                    noisemax = noise
                    meanmax = mean
        
            twd += 1 
            col1 += 1 # increment by 1
            if col1==ncol:
                col1 = 0 # If we're at the edge, go back to the begining
    
    return s2nmax,twdmax,colmax,meanmax,noisemax

# Computes SNR for every phase for a range of trial durations

cpdef bls2(np.ndarray[np.float64_t, ndim=1] data,
          np.ndarray[np.int64_t, ndim=1] mask, 
          np.ndarray[np.int64_t, ndim=1] col,
          twd1,
          twd2,):
    """
    """

    cdef int icad, icol,ncad, ncol,colmax,twdmax,col0
    cdef float s2nmax,
    
    ncad = data.shape[0]
    ncol = np.max(col)+1
    
    
    s2nmax = -1
    twdmax = -1
    colmax = -1
    for col0 in range(ncol):
        s2n,twd = blscol1(
            data,mask,col, col0, twd1, twd2,  ncol, ncad
        )
        
        if s2n > s2nmax:
            s2nmax = s2n
            twdmax = twd
            colmax = col0
    
    return s2nmax,twdmax,colmax

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef blscol1(np.ndarray[np.float64_t, ndim=1] data,
              np.ndarray[np.int64_t, ndim=1] mask, 
              np.ndarray[np.int64_t, ndim=1] col,
              int col0, int twd1, int twd2, int ncol, int ncad,):
   
    # Loop over cadencs
    cdef float c,s,ss,s2nmax,std,s2n,mean,noise
    cdef int twdmax,twd,i,icad,col1

    c = 0. # number of good data points
    s = 0. # sum of the data points
    ss = 0. # sum of squares of data

    s2nmax = 0.
    twdmax = 0
    twd = 1

    col1 = col0
    for i in range(twd2):
        for icad in range(ncad):
            # Increment counters
            if (mask[icad]==0) and (col[icad]==col1):
                c+=1 
                s+=data[icad] 
                ss+=data[icad]**2

        # Compute mean transit depth and scatter
        if c > 1:
            mean = s / c
            std = sqrt( (c*ss-s**2) / (c * (c - 1)))
            noise = std / sqrt(c)
            s2n = -1.0 * mean / noise

        if (s2n > s2nmax) and (twd >= twd1):
            s2nmax = s2n
            twdmax = twd
            
        
        twd += 1 
        col1 += 1 # increment by 1
        if col1==ncol:
            col1 = 0 # If we're at the edge, go back to the begining
    return s2nmax,twdmax

@cython.boundscheck(False)
def forman_mackey(float Pcad, 
                  float alpha, 
                  np.ndarray[np.int64_t, ndim=1] good_1d,
                  np.ndarray[DTYPE_t, ndim=1] dll_1d,
                  np.ndarray[DTYPE_t, ndim=1] depth_1d,
                  np.ndarray[DTYPE_t, ndim=1] depth_ivar_1d):
    """
    Evaluate Foreman-Mackey periodogram

    dll : delta log-likihood for single transit model
    depth_1d : depth of of individual transits
    """
    cdef int i, icol, ncol, ncad
    cdef double d

    ncad = depth_1d.size
    icad = np.arange(ncad)

    cdef np.ndarray[IDTYPE_t, ndim=1] col = \
            np.zeros( ncad, dtype=IDTYPE)
    cdef np.ndarray[IDTYPE_t, ndim=1] row = \
            np.zeros( ncad, dtype=IDTYPE)

    row,col = wrap_icad(icad,Pcad)
    nrow = np.max(row) + 1
    ncol = np.max(col) + 1

    # Allocate the output arrays.
    cdef np.ndarray[DTYPE_t, ndim=1] phic_same = \
            np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] phic_variable = \
            np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] depth_2d = \
            np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] depth_ivar_2d = \
            np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[IDTYPE_t, ndim=1] nind = \
            np.zeros( ncol, dtype=IDTYPE)

    
    cdef np.ndarray[IDTYPE_t, ndim=1] outcol = np.arange(ncol)

    # Loop over all the cadences
    for i in range(ncad):
        # If there is a problem with this particular transit, skip
        if good_1d[i]==0:
            continue

        icol = col[i]

        # First incorporate the delta log-likelihood for this
        # transit time.
        phic_variable[icol]+=dll_1d[i]

        # And then the uncertainty in the depth measurement.
        phic_variable[icol] += 0.5*log(depth_ivar_1d[i])

        # Here, we'll accumulate the weighted depth
        # measurement for the single depth model.
        depth_2d[icol] += depth_1d[i] * depth_ivar_1d[i]
        depth_ivar_2d[icol] += depth_ivar_1d[i]
        nind[icol]+=1

    for icol in range(ncol):
        # Penalize the PHICs for the number of free parameters.
        phic_same[icol] = phic_variable[icol] - 0.5 * alpha
        phic_variable[icol] -= 0.5 * nind[icol] * alpha
        depth_2d[icol] /= depth_ivar_2d[icol]

    for i in range(ncad):
        # Loop over the saved list of transit times and evaluate
        # the depth measurement at the maximum likelihood location.
        if good_1d[i]==0:
            continue

        icol = col[i]
        d = depth_1d[i] - depth_2d[icol] # difference between depths
        phic_same[icol] -= 0.5 * d * d * depth_ivar_1d[i]
        
    return outcol, phic_same, phic_variable, depth_2d, depth_ivar_2d, nind 


@cython.boundscheck(False)
def forman_mackey_max(float Pcad, 
                      float alpha, 
                      np.ndarray[IDTYPE_t, ndim=2] good_1d,
                      np.ndarray[DTYPE_t, ndim=2] dll_1d,
                      np.ndarray[DTYPE_t, ndim=2] depth_1d,
                      np.ndarray[DTYPE_t, ndim=2] depth_ivar_1d):

    """
    Run forman mackey (above) and maximize over durations
    """

    ntwd = good_1d.shape[0]
    ncad = good_1d.shape[1]

    icad = np.arange(ncad)
    row,col_temp = wrap_icad(icad,Pcad)
    cdef float max_phic_same, max_phic_variable, max_depth_2d, max_depth_ivar_2d, max_nind
    cdef int max_itwd, max_col
    cdef int icol,ncol, itwd
    ncol = np.max(col_temp)+1

    # Allocate the output arrays.
    cdef np.ndarray[IDTYPE_t, ndim=1] col = np.zeros(ncol, dtype=IDTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] phic_same = -np.inf + np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] phic_variable = -np.inf + np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] depth_2d = np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] depth_ivar_2d = np.zeros(ncol, dtype=DTYPE)
    cdef np.ndarray[IDTYPE_t, ndim=1] nind = np.zeros(ncol, dtype=IDTYPE)

    for itwd in range(ntwd):
        col, phic_same, phic_variable, depth_2d, depth_ivar_2d, nind = \
            forman_mackey(
                Pcad, 
                alpha, 
                good_1d[itwd], 
                dll_1d[itwd], 
                depth_1d[itwd], 
                depth_ivar_1d[itwd]
            )
        for icol in range(ncol):
            if depth_2d[icol] < 0.0:
                continue
            if phic_same[icol] > phic_variable[icol]:
                continue
            if phic_same[icol] > max_phic_same:
                max_phic_same = phic_same[icol]
                max_phic_variable = phic_variable[icol]
                max_depth_2d = depth_2d[icol]
                max_depth_ivar_2d = depth_ivar_2d[icol]
                max_nind = nind[icol]
                max_col = icol
                max_itwd = itwd          

    return max_phic_same, max_phic_variable, max_depth_2d, max_depth_ivar_2d, max_nind, max_col, max_itwd
