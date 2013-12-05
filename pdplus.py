"""
Module with routines to augment pandas library
"""

def LittleEndian(r):
    names = r.dtype.names
    data = {}
    for n in names:
        if r[n].dtype.byteorder=='>':
            data[n] = r[n].byteswap().newbyteorder() 
        else:
            data[n] = r[n] 
    q = pd.DataFrame(data,columns=names)
    return np.array(q.to_records(index=False))
