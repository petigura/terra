
def savunpack(data):
    """
    The idlsave python module returns a tuple where each element is a structure.
    However, one cannot access arrays within arrays as a 2 dim matrix.  This
    program unpacks that data.
    nxm (row x col)
    """
    import numpy as np

    n = len(data)
    m = len(data[0])
    datanew = np.empty((n,m))
    
    for i in range(n):
        for j in range(m):
            datanew[i][j] = data[i][j]

    return datanew
