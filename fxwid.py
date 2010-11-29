import numpy as np

def rdfixwid(file,colist,typelist,empstr=None,skiprows=0):
    """
    Read fixedwidth data and spit out arrays
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

def fxwd2rec(file,colist,typelist,empstr=None,skiprows=0):
    """
    Read fixedwidth data and spit out record

    Requires knowing the datatypes before runtime.
    file     - path to file 
    colist   - [ [<begin col>,<end col>], [<begin col>,<end col>], ... ]
    typelist - type format ala np.array
               [ ('name','|S10'), ('number',float)]
    empstr   - string to interpreate as None e.g. '---'
              
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


def emp2none(inpstr,empstr):
    """
    If inpstr corresponds to an empty string.  Return a none. 
    """

    if inpstr.strip() == empstr:
        inpstr = None

    return inpstr
