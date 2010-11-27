class Getelnum:
    def __init__(self,line):
        # Parameters specific to both lines
        self.teff_sol = 5770        # Solar Effective Temp
        self.tefferr  = 44          # FV05 estimated error in Temp
        self.feherr = 0.03          # FV05 estimate for error in [Fe/H]

        self.scattercut = 0.3       # Cut on the scatter of derived fits
        self.coThresh = 1  #theshold (high) value of C/O

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
            self.abnd_sol_luck = 8.75
            self.err_luck = 0.05

            # Plotting & Preprocessing parameters (might be unessasary)
            self.wran = [6295,6305]
            self.sran = [6299.3,6301.]
            self.yrg  = [.9,1.02]
            self.sampxr = [6299.9,6300.6]
            self.chip = 'r'
            self.boolidiv = 1
            self.booltell = 1
            self.ord = 14
            self.chirng = [6300.000,6300.600]

        # Parameters specific to Carbon line
        if line == 6587 or line == 'C':
            # Line IDs
            self.elnum = 6
            self.line  = 6587
            self.elstr = 'C'
            self.abnd_sol = 8.5

            # Cut limits
            self.vsinicut = 15         # max vsini allowed
            self.teffrng = [5300,6500] # Permitted range of Teff

            # Comparison Data
            self.abnd_sol_luck = 8.54
            self.err_luck = 0.05

            # Plotting & Preprocessing parameters (might be unessasary)
            self.wran = [6584,6591]
            self.sran = [6587.2,6588]
            self.chirng = [6587.4,6587.8]
            self.yrg  = [.9,1.02]
            self.sampxr = [6587.2,6588.]            
            self.chip = 'i'
            self.boolidiv = 0
            self.booltell = 0
            self.ord = 0
            self.tellrng = [6587.4,6587.8]
            
        # Field IDs
        lowelstr = self.elstr.lower()
        self.abundfield = lowelstr+'_abund'
        self.staterrfield = lowelstr+'_staterr'



