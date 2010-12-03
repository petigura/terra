"""
Module for interacting with latex
"""
import numpy as np

def flt2tex(fnum,sigfig=3):
    logfnum = np.log10(fnum)
    if logfnum > 0:
        exp = int(logfnum)
    else:
        exp = int(logfnum-1)

    mantissa = fnum/10**exp
    beg  = r'$ %.'+str(sigfig-1)+'f'
    beg  = beg % mantissa
    end  = r' \times 10^{%i} $' % exp
    return beg+end
