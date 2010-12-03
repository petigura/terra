"""
Functions to parse fixed width data.

fxwd2rec - Converts fixed width data and returns record array.  
           Types are specified a la np.array format e.g.
           [ ('name','|S10'), ('number',float)]

fxwd2tup - Same as above but returns tuple of arrays
"""
import numpy as np

def fxwd2rec(file,colist,typelist,empstr=None,skiprows=0):
    """
    Read fixed width data and return record array

    Requires knowing the datatypes before runtime.
    file     - path to file 
    colist   - [ [<begin col>,<end col>], [<begin col>,<end col>], ... ]
    typelist - type format ala np.array
               [ ('name','|S10'), ('number',float)]
    empstr   - string to interperate as None e.g. '---'              
    """
    outlist =[]
    f = open(file,'r')
    txt = f.readlines()
    txt = txt[skiprows:]
    nrows = len(txt)
    ncol  = len(colist)

    outrec = np.array([], dtype=typelist)

    for i in range(nrows):

        # Build up each line
        row = []
        for j in range(ncol):
            word = txt[i][colist[j][0]:colist[j][1]]
            if empstr is not None:
                word = emp2none(word,empstr)
            row.append(word)

        rowtup = tuple(row)
        rowrec = np.array(rowtup,dtype=typelist)
        outrec = np.append(outrec,rowrec)            
        
    return outrec

def fxwd2tup(file,colist,typelist,empstr=None,skiprows=0):
    """
    Read fixed width data and returns tuple of arrays.

    Requires knowing the datatypes before runtime.
    file     - path to file 
    colist   - [ [<begin col>,<end col>], [<begin col>,<end col>], ... ]
    typelist - type format ala np.array
               [ ('name','|S10'), ('number',float)]
    empstr   - string to interperate as None e.g. '---'
    """
    outlist =[]
    f = open(file,'r')
    txt = f.readlines()
    txt = txt[skiprows:]
    nrows = len(txt)
    ncol  = len(colist)

    for type in typelist:
        outlist.append(np.zeros(nrows,dtype=type))
                
    for i in range(nrows):
        for j in range(ncol):
            word = txt[i][colist[j][0]:colist[j][1]]
            if empstr is not None:
                word = emp2none(word,empstr)
            outlist[j][i] = word

    return tuple(outlist)


def emp2none(inpstr,empstr):
    """
    If inpstr corresponds to an empty string.  Return a none. 
    """

    if inpstr.strip() == empstr:
        inpstr = None

    return inpstr
