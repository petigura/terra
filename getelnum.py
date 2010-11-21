class Getelnum:
    def __init__(self,line):
        self.teff_sol = 5770
        self.scattercut = 0.3
        self.coThresh = 1  #theshold (high) value of C/O
        
        if line == 6300 or line == 'O':
            self.elnum=8.
            self.line = 6300
            self.wran = [6295,6305]
            self.sran = [6299.3,6301.]
            self.yrg  = [.9,1.02]
            self.sampxr = [6299.9,6300.6]
            self.chip = 'r'
            self.boolidiv = 1
            self.booltell = 1
            self.ord = 14
            self.chirng = [6300.000,6300.600]
            self.elstr = 'O'
            self.abnd_sol = 8.7 #solar abnd of oself.ygen
            self.vsinicut = 7
        
            self.abnd_sol_luck = 8.75
            self.err_luck = 0.05
            self.teffrng = [4700,6500]


        if line == 6587 or line == 'C':
            self.elnum=6
            self.line = 6587
            self.wran = [6584,6591]
            self.sran = [6587.2,6588]
            self.chirng = [6587.4,6587.8]
            self.yrg  = [.9,1.02]
            self.sampxr = [6587.2,6588.]
            
            self.chip = 'i'
            self.boolidiv = 0
            self.booltell = 0
            self.ord = 0
            
            self.elstr = 'C'
            self.tellrng = [6587.4,6587.8]
            self.abnd_sol = 8.5
            # Carbon line is farther from other lines, so it can be broader
            self.vsinicut = 15

            self.teffrng = [5300,6500]

            self.abnd_sol_luck = 8.54
            self.err_luck = 0.05
            
        lowelstr = self.elstr.lower()
        self.abundfield = lowelstr+'_abund'
        self.staterrfield = lowelstr+'_staterr'
        self.feherr = 0.03
        self.tefferr= 44
