import pandas as pd

class HDFStore(object):
    """Store objects to HDF5 data format

    Args:
         None

    Attributes:
         header (pandas DataFrame): list of header values and descriptions
         datasets (pandas DataFrame) : list of datasets
    """

    def __init__(self):
        self.header = pd.DataFrame(index=[],columns=['value','description'])
        self.header.index.name = 'name'

        self.tables = pd.DataFrame(index=[],columns=['shape','description'])
        self.tables.index.name = 'name'

    def update_header(self, *args):
        """ Update header information, create if it doesn't exist 
        
        Args:
            name (str) : name of header attribute
            value : value of header attribute
            description : short description of value
        """
        name = args[0]
        value = args[1]
        setattr(self, name, value)

        if len(args)==3:
            description = args[2]
            self.header.loc[name] = [value, description]
        elif len(args)==2:
            self.header.loc[name,'value'] = value

    def update_table(self, *args):
        """ Update table information, create if it doesn't exist.
        
        Args:
            name (str) : name of header attribute
            table (pandas DataFrame) : table data 
            description : short description of value
        """
        name = args[0]
        table = args[1]
        setattr(self, name, table)

        if len(args)==3:
            description = args[2]
            self.tables.loc[name] = [table.shape, description]
        elif len(args)==2:
            self.tables.loc[name,'shape'] = table.shape

    def info(self):
        """ Print info about the object
        """
        
        print "Header:"
        print self.header.to_string()
        
        print 
        print "Tables:"
        print self.tables.to_string()
        

    def to_hdf(self, h5file, group):
        """ Write object as HDF5 file

        Args:
            h5file (str): path to h5file
            group (str): base group to write to. If set to '/', an object with 
                tableA will write to '/header', '/tableA'. If set to '/group',
                will write to '/group/header', '/group/tableA'
        """
        
        self.header.to_hdf(h5file, group + '/header')
        self.tables.to_hdf(h5file, group + '/tables')
        for table_name, row in self.tables.iterrows():
            table = getattr(self, table_name)
            table.to_hdf(h5file, group + '/' + table_name)

    def read_hdf(self, h5file, group):
        """ Read info from HDF file """
        self.header = pd.read_hdf(h5file, group + '/header')
        self.tables = pd.read_hdf(h5file, group + '/tables')

        for table_name, row in self.tables.iterrows():
            table = pd.read_hdf(h5file, group + '/' + table_name)
            setattr(self, table_name, table)

        for name, row in self.header.iterrows():
            setattr(self, name, row['value'])
