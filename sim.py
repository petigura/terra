"""
8669806|17.0088653564453 <- Studied this one before.
8144222|20.9299125671387
8212005|30.5336647033691
8410490|40.3160285949707
"""
import atpy
import platform
import os
import stat
from numpy import ma
import glob
import subprocess
import numpy as np
import copy

import qalg
import tfind
import ebls
import git
import tval
import keptoy

def makeGridScript(LCfile,PARfile,nsim=None,view=None,test=False):
    simpath = os.path.dirname(LCfile)

    tPAR = atpy.Table( PARfile )
    if test:
        Psmp = 4
        nsim = 1
        suffix = '_test'
    else:
        Psmp = 0.25
        nsim = len(tPAR.data)
        suffix = ''


    scripts = []
    for isim in range(nsim):
        template = """
import sim
sim.grid( '%(LCfile)s','%(PARfile)s' , %(isim)i, Psmp = %(Psmp)f  )
""" % {'LCfile':LCfile,'PARfile':PARfile,'isim':isim,'Psmp':Psmp}

        fpath = os.path.join(simpath,'grid%04d%s.py' % (isim,suffix)   )
        fpath = os.path.abspath(fpath)

        
        f = open( fpath,'wb' )
        f.writelines(template)
        f.close()
        
        os.chmod(fpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        scripts.append(fpath)
        
    print template

    return scripts
    
def grid(LCfile,PARfile,seed,Psmp=0.25):
    simpath = os.path.dirname(LCfile)
    tLC =  atpy.Table( LCfile )
    tPAR = atpy.Table( PARfile )

    f = keptoy.genEmpLC( qalg.tab2dl( tPAR.rows([seed]) )[0] , tLC.t , tLC.f)
    t = tLC.t

    PG0 = ebls.grid( tLC.t.ptp() , 0.5, Pmin=50.0, Psmp=Psmp)
    res = tfind.tfindpro(t,f,PG0,0)
    tRES = qalg.dl2tab([res])

    tRES.add_keyword("LCFILE" ,LCfile)
    tRES.add_keyword("PARFILE",PARfile)
    tRES.table_name = "RES"
    tRES.comments = "Table with the simulation results"
    repo = git.Repo('/Users/petigura/Marcy/Kepler/pycode/')
    headcommit = repo.head.commit
    tRES.add_keyword( 'GitCommit' , headcommit.hexsha)
    tRES.add_keyword( 'RepoDirty' , repo.is_dirty() )
    tRES.add_keyword( 'Node'      , platform.node()  )
    RESfile = os.path.join(simpath,'tRES%04d.fits' % seed)
    tRES.write(RESfile,overwrite=True)

def makeValScript(LCfile,PARfile,nsim=None,test=False):
    tPAR = atpy.Table( PARfile )
    simpath = os.path.dirname(LCfile)

    if test:
        nCheck = 5
        nsim = 1
        suffix = '_test'
    else:
        nCheck = 50
        nsim = len(tPAR.data)
        suffix = ''

    scripts = []
    for seed in range(nsim):
        template = """
import sim
sim.val('%(LCfile)s','%(PARfile)s',%(nCheck)i, %(seed)i )
""" % {'LCfile':LCfile,'PARfile':PARfile,'seed':seed,'nCheck':nCheck}

        fpath = os.path.join(simpath,'val%04d%s.py' % (seed,suffix)   )
        f = open( fpath,'wb' )
        f.writelines(template)
        os.chmod(fpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        scripts.append(fpath)

    print template

    return scripts


def val(LCfile,PARfile,nCheck,seed):
    simpath = os.path.dirname(LCfile)

    tLC =  atpy.Table( LCfile )
    tPAR = atpy.Table( PARfile )
    f = keptoy.genEmpLC( qalg.tab2dl( tPAR.rows([seed]) )[0] , tLC.t , tLC.f)
    t = tLC.t

    RESfile = os.path.join(simpath,'tRES%04d.fits' % seed)
    tRES = atpy.Table(RESfile)

    dL = tval.parGuess(qalg.tab2dl(tRES)[0],nCheck=nCheck)

    print 21*"-" + " %d" % (seed)
    
    resL = tval.fitcandW(t,f,dL)
    print "   iP      oP      s2n    "
    for d,r in zip(dL,resL):
        print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])


    # Alias Lodgic.
    resL = [r for r in resL if  r['s2n'] > 5]

    resL = tval.aliasW(t,f,resL)
    for d,r in zip(dL,resL):
        print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])

    tVAL = qalg.dl2tab(resL)
    tVAL.keywords = tRES.keywords
    tVAL.keywords['RESFILE'] = RESfile

    VALfile = os.path.join(simpath,'tVAL%04d.fits' % seed )
    tVAL.write(VALfile,overwrite=True)

def simSetup():
    """
    Walk user through setting up a script.
    """

    simpath = raw_input('Enter Simulation Directory: ')
    simpath = os.path.abspath(simpath)

    LCfile = os.path.join(simpath,'tLC.fits')
    PARfile = os.path.join(simpath,'tPAR.fits')

    tPAR = atpy.Table(PARfile)
    tLC  = atpy.Table(LCfile)

    gridScripts = makeGridScript(LCfile,PARfile)
    valScripts  = makeValScript(LCfile,PARfile)

    gridScriptTest = makeGridScript(LCfile,PARfile,test=True)
    valScriptTest  = makeValScript(LCfile,PARfile,test=True)

    boolRunGrid = yesno( raw_input('Process Grid Scripts? [y/n]') )
    boolRunVal  = yesno( raw_input('Process Val Scripts? [y/n] ')  )

    from IPython.parallel import Client
    rc = Client()
    view = rc.load_balanced_view()

    runfail = True
    while runfail:
        print "Running a test case"
        bgrid = view.map(srun,gridScriptTest,block=True)[0]
        bval  = view.map(srun,valScriptTest,block=True)[0]
        print "Test Grid Exit Status %i" % bgrid
        print "Test Val Exit Status %i" % bval

        runfail = bgrid | bval # Both exit statuses must be 0
        if ~runfail:
            print "both test runs suceeded"
        

    RESfiles = glob.glob( os.path.join(simpath,'tRES????.fits') )
    VALfiles = glob.glob( os.path.join(simpath,'tVAL????.fits') )

    if ~yesno( raw_input('Overwrite tRES ? [y/n]') ):
        for RESfile in RESfiles:
            f = os.path.join(simpath,'grid%04d.py' % name2seed(RESfile) )
            gridScripts.remove(f)

    if ~yesno( raw_input('Overwrite tVAL ? [y/n]') ):
        for VALfile in VALfiles:
            f = os.path.join(simpath,'val%04d.py' % name2seed(VALfile) )
            valScripts.remove(f)

    if boolRunGrid:
        view.map(srun,gridScripts,block=True)

    if boolRunVal:
        view.map(srun,valScripts,block=True)
        files = glob.glob(simpath+'tRES????.fits')
        tRED = resRed(tPAR,files)
        tRED.write(os.path.join(simpath,'tRED.fits'))

def srun(s):
    """
    Convert a script to a python call + log
    """
    log = s.split('.')[0]+'.log'
    return subprocess.call( 'python %s > %s' % (s,log) ,shell=True )


def iPoP(tval):
    """
    """
    s2n = ma.masked_invalid(tval.s2n)
    iMax = s2n.argmax()
    t = tval.rows( [iMax] )
    return t

def yesno(s):
    """
    Convert yes no to boolean
    """
    if s[0] == 'y':
        return True
    else:
        return False

def diagFail(t):
    """
    Why did a particular run fail?
    """
    kepdir = os.environ['KEPDIR']
    Palias = t.P[0] * np.array([0.5,2])

    # Likely explaination for failure : alias function.
    balias = (abs( t.oP[0] / Palias - 1) < 1e-3).any()
    tLC  = atpy.Table(t.keywords['LCFILE'],type='fits')
    pknown = qalg.tab2dl(t)[0]
    f = keptoy.genEmpLC(pknown , tLC.t,tLC.f  )
    dM,bM,aM,DfDt,f0 = tfind.MF(f,20)
    Pcad = round(pknown['P']/keptoy.lc)
    res = tfind.ep(dM,Pcad)
    win = res['win']
    # Likely explaination for failure : window function.
    bwin = ~(win[np.floor(t.epoch[0]/keptoy.lc)]).astype(bool)
    
    return balias,bwin


def addFlag(t):
    """
    Adds a string description as to why the run failed'
    """

    baliasL,bwinL = [],[]
    for i in range(len(t.data)):
        balias,bwin = diagFail(t.rows([i]))
        baliasL.append(balias)
        bwinL.append(bwin)
    keplerio.update_columns(t,'balias',baliasL)
    keplerio.update_columns(t,'bwin',bwinL)
    return t


def kwUnique(kwL,key):
    """
    Confirm that all instances of keyword are unique.  Return that unique value
    """
    valL = np.unique( np.array( [ kw[key] for kw in kwL ]  ) )
    assert valL.size == 1, '%s must be unique' % key
    return valL[0]

def resRed(files,PARfile=None,LCfile=None):
    """
    Result Reduce.

    Read in all the tVAL results and augment tPAR with the results.

    Parameters
    ----------
    files : A list of tVAL to reduce.
    """

    # Build table containing input parameters.
    seedL,parfileL=[],[]

    tVALL = [atpy.Table(f,type='fits') for f in files]

    if (PARfile == None) | (LCfile == None):
        kwL = [t.keywords for t in tVALL ]

    if PARfile == None:
        PARfile = kwUnique(kwL,'PARFILE')

    if PARfile == None:
        LCfile = kwUnique(kwL,'LCFILE')

    tPAR = atpy.Table(PARfile,type='fits')

    for f in files:
        bname = os.path.basename(f)
        bname = os.path.splitext(bname)[0]
        seed = int(bname.split('tVAL')[-1])
        seedL.append(seed)

    tRED = tPAR.rows(seedL)
    tRED.keywords = tVALL[0].keywords
    
    ttemp = iPoP( atpy.Table(files[0],type='fits') )
    for f in files[1:]:
        tnew = iPoP( atpy.Table(f,type='fits') )
        ttemp.append( tnew )

    col = ['s2n','P','epoch','tdur','df'] # columns to attach
    for c in col:
        keplerio.update_columns(tRED,'o'+c,ttemp[c])    

    addbg(tRED)
    addFlag(tRED)
    return tRED

def addbg(t):    
    bg = ( abs(t.P - t.oP)/t.P < 0.001 ) & ( abs(t.epoch - t.oepoch) < 0.1 )
    keplerio.update_columns(t,'bg',bg)    
    return t

def getkw(file):
    """
    Open all the files.  Return list of KW dictionaries.
    """    
    t = atpy.Table(file)
    return t.keywords

def getkwW(files,view=None):
    if view==None:
        return map(getkw,files)
    else:
        return view.map(getkw,files)

def quickRED(RESfiles,view=None):
    """
    Quick look at the periodogram.  Do not incorporate tVAL information.
    """

    kwL = getkwW(RESfiles,view=view)
    PARfileL = [kw['PARFILE'] for kw in kwL]
    PARfile = np.unique(PARfileL)
    LCfileL = [kw['LCFILE'] for kw in kwL]
    LCfile = np.unique(LCfileL)
    assert PARfile.size == 1, "PARfile must be unique"
    assert PARfile.size == 1, "LCfile must be unique"

    PARfile = PARfile[0]
    LCfile = LCfile[0]

    tRED = atpy.Table(PARfile,type='fits')
    tRED.keywords['PARFILE'] = PARfile
    tRED.keywords['LCFILE'] = LCfile
    
    col = ['P','epoch'] # columns to attach
    ocol = ['o'+c for c in col]
    for o in ocol:
        tRED.add_empty_column(o,np.float)
    
    for RESfile in RESfiles:
        tRES = atpy.Table(RESfile,type='fits')
        seed = name2seed(RESfile)

        dL = tval.parGuess( qalg.tab2dl(tRES)[0] )
        d = dL[0]
        irow = np.where(tRED.seed == seed)[0]

        for c,o in zip(col,ocol):
            tRED[o][irow] = d[c]

    addbg(tRED)
    addFlag(tRED)
    return tRED

def name2seed(file):
    """
    Convert the name of a file to a seed value.
    """
    bname = os.path.basename(file)
    bname = os.path.splitext(bname)[0]
    return int(bname[-4:])
    
def addVALkw(files):
    """
    Add input information to the tVAL files
    """

    tVALL = [atpy.Table(f,type='fits') for f in files]
    kwL = [t.keywords for t in tVALL ]

    PARfile = kwUnique(kwL,'PARFILE')
    tPAR = atpy.Table(PARfile,type='fits')

    for t,f in zip(tVALL,files):
        seed = name2seed(f)
        tROW = tPAR.where(tPAR.seed == seed)
        assert tROW.data.size == 1, 'Seed must be unique'
        col = ['P','epoch','df','tdur','seed','tbase']
        for c in col:
            t.keywords[c] = tROW.data[c][0]
        t.write(f,overwrite=True,type='fits')

    
