import idlsave
import numpy as np

class idlobj:
    def __init__(self,savfile,var):
        """
        Creates a python class from an IDL array of structures.

        savfile - Path to the IDL .sav file.
        var     - string. IDL .sav files can store multiple variables.
                  Specify the variable to be converted.

        ex:
        stars = PyDL.idlobj(os.environ['PYSTARS'],'stars')
        """

        idldict = idlsave.read(savfile) # dict of idlvariables
        self.arr  = idldict[var]        # assume variable is an array
        self.el0 = self.arr[0]          # first element is template
        self.fields = np.array(self.arr.dtype.names)
        
        #make fields lowercase
        for i in range(len(self.fields)):
            self.fields[i] = self.fields[i].lower()

        #assign attributes to class
        for field in self.fields:
            if type(self.el0[field]) is np.ndarray:
                unpackedfield = savunpack(self.arr[field])
                exec('self.'+field+' = unpackedfield')
            else:
                exec('self.'+field+' = self.arr["'+field+'"]')

def savunpack(data):
    """    
    The idlsave python module maps an idl structure to a record array.
    Arrays of structures that contain arrays are mapped to arrays of arrays.
    This function "unpacks" those arrays into multidimensional arrays.

    IDL> help,stars     
    STARS           STRUCT    = -> <Anonymous> Array[1070]

    IDL> help,stars,/str                  
    ** Structure <2282e04>, 61 tags, length=744, data length=735, refs=1:
    NAME            STRING    '4915'
    O_STATERR       FLOAT     Array[2]


    In python: 
    >>> test['stars']['name'],test['stars']['O_STATERR']
    (array([4915, ..., GANYMEDE], dtype=object),
     array([[-0.04204083  0.04433005], ... ,
            [-1.40295613  0.1279441 ], dtype=object))

    
    """
    import numpy as np

    n = len(data)
    m = len(data[0])
    datanew = np.empty((n,m))
    
    for i in range(n):
        for j in range(m):
            datanew[i][j] = data[i][j]

    return datanew
