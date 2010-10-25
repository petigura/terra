import numpy as np

def recunpack(rec):
    fields = rec.dtype.names
    nrec = len(rec)
    temprec = copy(rec)
    for field in fields:
        if type(newrec[field]) is not np.ndarray:
            rec = mlab.rec_drop_fields(rec,field)


    newrec = rec[0]
    
    for field in fields:
        
        if type(newrec[field]) is np.ndarray:
            l = len(newrec[field])
            newrec[field] = resize(newrec[field],(nrec,l))
            #fill in other elements
            for i in np.range(nrec):
                newrec[field][i] = rec[field][i]
        else:                
            
