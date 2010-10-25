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



def emp2none(inpstr,empstr):
    """
    If inpstr corresponds to an empty string.  Return a none. 
    """
    if inpstr.strip() is empstr:
        inpstr = None

    return inpstr
