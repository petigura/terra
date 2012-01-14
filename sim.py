"""
8669806|17.0088653564453 <- Studied this one before.
8144222|20.9299125671387
8212005|30.5336647033691
8410490|40.3160285949707
"""
import atpy
import platform

import keplerio
import qalg
import tfind
import ebls
import git
import os
import tval

KIC  = 8144222

simpath = 'Sim/01-13-12/KIC-%09d/' % KIC
simpath = os.path.join(os.environ['KEPDIR'],simpath)

def gridW(LCfile,PARfile,nsim=None,view=None,test=False):
    tLC = atpy.Table(LCfile)
    tPAR = atpy.Table(PARfile)
    
    if nsim==None:
        nsim = len(tPAR.data)

    PG0 = ebls.grid( tPAR.tbase[0] , 0.5, Pmin=50.0, Psmp=0.25)
    if test:
        PG0 = PG0[::10]

    func = lambda i: grid(tLC.t,tLC.f,tPAR,PG0,i)

    irng = range(nsim)
    if view==None:
        map(func,irng)
    else:
        view.map( grid,nsim*[tLC.t],nsim*[tLC.f],nsim*[tPAR],nsim*[PG0], irng,block=True)

def grid(t,f,tPAR,PG0,i):
    t,f = qalg.genEmpLC( [qalg.tab2dl(tPAR)[i]] , t , f )
    res = tfind.tfindpro(t[0],f[0],PG0,0)
    tres = qalg.dl2tab([res])
    tres.table_name = "RES"
    tres.comments = "Table with the simulation results"
    repo = git.Repo('/Users/petigura/Marcy/Kepler/pycode/')
    headcommit = repo.head.commit
    tres.add_keyword( 'GitCommit' , headcommit.hexsha)
    tres.add_keyword( 'RepoDirty' , repo.is_dirty() )
    tres.add_keyword( 'Node'      , platform.node()  )

    RESfile = os.path.join(simpath,'tRES%04d.fits' % i)
    tres.write(RESfile,overwrite=True)

def valW(LCfile,PARfile,nsim=None,view=None,test=False):
    tLC = atpy.Table(LCfile)
    tPAR = atpy.Table(PARfile)
    
    if test:
        nCheck = 5
    else:
        nCheck = 50

    tl,fl = qalg.genEmpLC( qalg.tab2dl(tPAR) , tLC.t , tLC.f)

    if nsim==None:
        nsim = len(tPAR.data)

    tl,fl = tl[:nsim],fl[:nsim]

    func = lambda t,f,i: val(t,f,nCheck,i)

    if view==None:
        map(func,tl,fl,range(nsim))
    else:
        view.map(val,nsim*[tl],nsim*[fl],nsim*[nCheck],range(nsim),block=True)


def val(t,f,nCheck,i):
    tRES = atpy.Table(os.path.join(simpath,'tRES%04d.fits' % i))
    dL = tval.parGuess(qalg.tab2dl(tRES)[0],nCheck=nCheck)
    resL = tval.fitcandW(t,f,dL)

    print 21*"-" + " %d" % (i)
    print "   iP      oP      s2n    "
    for d,r in zip(dL,resL):
        print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])
    tVAL = qalg.dl2tab(resL)
    VALfile = os.path.join(simpath,'tVAL%04d.fits' % i)
    tVAL.write(VALfile,overwrite=True)
