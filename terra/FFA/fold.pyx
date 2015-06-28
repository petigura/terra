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
