class Getelnum:
    def __init__(self,line):
        # Parameters specific to both lines
        self.teff_sol = 5770        # Solar Effective Temp
        self.tefferr  = 44          # FV05 estimated error in Temp
        self.feherr = 0.03          # FV05 estimate for error in [Fe/H]

        self.scattercut = 0.3       # Cut on the scatter of derived fits
        self.coThresh = 1  #theshold (high) value of C/O

        self.lines = [6300,6587]
        self.elements = ['O','C']
        self.nel = len(self.lines) 

        # Parameters specific to Oxygen line
        if line == 6300 or line == 'O':
            # Line IDs
            self.elnum=8.
            self.line = 6300
            self.elstr = 'O'
            self.abnd_sol = 8.7 

            # Cut limits
            self.teffrng = [4700,6500] # max vsini allowed
            self.vsinicut = 7          # Permitted range of Teff

            # Comparison Info
            self.comptable  = 'ben05'
            self.compoffset = 0.
            self.compref    = 'Bensby 2005'
            self.comperr    = 0.06   

        # Parameters specific to Carbon line
        elif line == 6587 or line == 'C':
            # Line IDs
            self.elnum = 6
            self.line  = 6587
            self.elstr = 'C'
            self.abnd_sol = 8.5

            # Cut limits
            self.vsinicut = 15         # max vsini allowed
            self.teffrng = [5300,6500] # Permitted range of Teff

            # Comparison Data
            self.comptable   = 'luckstars'
            self.compoffset  = 8.5
            self.compref     = 'Luck 2006'
            self.comperr     = 0.1

        else:
            self.elstr =  None

        if self.elstr is not None:
            # Field IDs
            lowelstr = self.elstr.lower()
            self.abundfield = lowelstr+'_abund'
            self.staterrfield = lowelstr+'_staterr'



