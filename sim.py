"""
8669806|17.0088653564453 <- Studied this one before.
8144222|20.9299125671387
8212005|30.5336647033691
8410490|40.3160285949707
"""
import atpy
import platform

import qalg
import tfind
import ebls
import git
import os
import tval
import stat
from numpy import ma

KIC  = 8144222

simpath = 'Sim/01-13-12/KIC-%09d/' % KIC
simpath = os.path.join(os.environ['KEPDIR'],simpath)

def makeGridScript(LCfile,PARfile,nsim=None,view=None,test=False):

    tPAR = atpy.Table( PARfile )
    if nsim==None:
        nsim = len(tPAR.data)

    scripts = []
    for isim in range(nsim):
        template = """
import sim
sim.grid( '%(LCfile)s','%(PARfile)s' , %(isim)i  )
""" % {'LCfile':LCfile,'PARfile':PARfile,'isim':isim}

        fpath = os.path.join(simpath,'grid%04d.py' % isim   )
        fpath = os.path.abspath(fpath)

        f = open( fpath,'wb' )
        f.writelines(template)
        os.chmod(fpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        scripts.append(fpath)
        
    print template

    return scripts
    
def grid(LCfile,PARfile,seed):
    tLC =  atpy.Table( LCfile )
    tPAR = atpy.Table( PARfile )

    t,f = qalg.genEmpLC( [ qalg.tab2dl(tPAR)[seed] ] , tLC.t , tLC.f )
    PG0 = ebls.grid( tPAR.tbase[0] , 0.5, Pmin=50.0, Psmp=0.25)

    res = tfind.tfindpro(t[0],f[0],PG0,0)
    tres = qalg.dl2tab([res])
    tres.table_name = "RES"
    tres.comments = "Table with the simulation results"
    repo = git.Repo('/Users/petigura/Marcy/Kepler/pycode/')
    headcommit = repo.head.commit
    tres.add_keyword( 'GitCommit' , headcommit.hexsha)
    tres.add_keyword( 'RepoDirty' , repo.is_dirty() )
    tres.add_keyword( 'Node'      , platform.node()  )



    RESfile = os.path.join(simpath,'tRES%04d.fits' % seed)
    tres.write(RESfile,overwrite=True)

def makeValScript(LCfile,PARfile,nsim=None,view=None,test=False):
    tPAR = atpy.Table( PARfile )
    if test:
        nCheck = 5
    else:
        nCheck = 50

    if nsim==None:
        nsim = len(tPAR.data)

    scripts = []
    for seed in range(nsim):
        template = """
import sim
sim.val('%(LCfile)s','%(PARfile)s',%(nCheck)i, %(seed)i )
""" % {'LCfile':LCfile,'PARfile':PARfile,'seed':seed,'nCheck':nCheck}

        fpath = os.path.join(simpath,'val%04d.py' % seed   )
        f = open( fpath,'wb' )
        f.writelines(template)
        os.chmod(fpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        scripts.append(fpath)

    print template

    return scripts


def val(LCfile,PARfile,nCheck,seed):
    tLC =  atpy.Table( LCfile )
    tPAR = atpy.Table( PARfile )
    tl,fl = qalg.genEmpLC( qalg.tab2dl(tPAR) , tLC.t , tLC.f)
    t,f = tl[ seed],fl[ seed ]

    tRES = atpy.Table(os.path.join(simpath,'tRES%04d.fits' % seed))
    dL = tval.parGuess(qalg.tab2dl(tRES)[0],nCheck=nCheck)
    resL = tval.fitcandW(t,f,dL)

    print 21*"-" + " %d" % (seed)
    print "   iP      oP      s2n    "
    for d,r in zip(dL,resL):
        print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])

    tVAL = qalg.dl2tab(resL)
    VALfile = os.path.join(simpath,'tVAL%04d.fits' % seed )
    tVAL.write(VALfile,overwrite=True)

def resRed(PARfile,files):
    """
    Read out the results of the RES????.fits files.
    """
    
    # Build table containing input parameters.
    seedL=[]
    tPAR = atpy.Table(PARfile,type='fits')
    for f in files:
        bname = os.path.basename(f)
        bname = os.path.splitext(bname)[0]
        seed = int(bname.split('tVAL')[-1])
        seedL.append(seed)

    tPAR = tPAR.rows(seedL)
    
    ttemp = iPoP( atpy.Table(files[0],type='fits') )
    for f in files[1:]:
        tnew = iPoP( atpy.Table(f,type='fits') )
        ttemp.append( tnew )


    col = ['s2n','P','epoch'] # columns to attach
    for c in col:
        tPAR.add_column('o'+c,ttemp[c])
    
    return tPAR

def bg(tres):    
    bg = ( abs(tres.P - tres.oP)/tres.P < 0.001 ) & ( abs(tres.epoch - tres.oepoch)/tres.epoch < 0.1 )
    tres.add_column('bg',bg)
    return tres

def iPoP(tval):
    """
    """
    s2n = ma.masked_invalid(tval.s2n)
    iMax = s2n.argmax()
    t = tval.rows( [iMax] )
    return t

