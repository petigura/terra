import sqlite3
import numpy as np
import getelnum
import pdb
import uncertainties
from uncertainties import unumpy
import os
import postfit

def dump_stars(save=True,texcmd=False):
    conn = sqlite3.connect(os.environ['STARSDB'])
    cur = conn.cursor()

    elements = ['O','C']
    cuts={}
    texdict = {}
    cuts['C'] = postfit.globcut('C',table='t3')
    cuts['O'] = postfit.globcut('O',table='t2')


    cmd0 = """
    SELECT
    t1.name,t1.vmag,t1.d,t1.teff,t1.logg,t1.pop_flag,t1.monh,t1.ni_abund,
    t2.o_nfits,t2.o_abund,t2.o_staterrlo,t2.o_staterrhi,
    t3.c_nfits,t3.c_abund,t3.c_staterrlo,t3.c_staterrhi
    FROM 
    mystars t1

    LEFT JOIN
    mystars t3 
    ON
    t1.id = t3.id AND (%s)

    LEFT JOIN
    mystars t2 
    ON
    t1.id = t2.id AND (%s)

    WHERE (%s) OR (%s)

    ORDER BY 
    t1.name
    """ % (cuts['C'],cuts['O'],cuts['C'],cuts['O'])

    temptype = [('name','|S10'),('vmag',float),('d',int),('teff',int),
                ('logg',float),('pop_flag','|S10'),('monh',float),('ni_abund',float),
                ('o_nfits',float),('o_abund',float),('o_staterrlo',float),('o_staterrhi',float),
                ('c_nfits',float),('c_abund',float),('c_staterrlo',float),('c_staterrhi',float)]


    cur.execute(cmd0)
    out = cur.fetchall()#,dtype=temptype)
    
    out = np.array(out,dtype=temptype)
    outstr = []
    
    out['o_nfits'][np.where(np.isnan(out['o_nfits']))[0]]  = 0
    out['c_nfits'][np.where(np.isnan(out['c_nfits']))[0]]  = 0
    
    #subtract of solar abundance
    elements = ['O','C']
    abund,errhi,errlo,abndone,logeps = {},{},{},{},{}

    c2oarr = np.array([])

    for el in elements:
        p = getelnum.Getelnum(el)        
        elo = el.lower()
        abund[el] = out['%s_abund' % elo] - p.abnd_sol
        logeps[el] = out['%s_abund' % elo]
        errhi[el] = np.abs(out['%s_staterrlo' % elo])
        errlo[el] = out['%s_staterrhi' % elo]
    
    for i in range(len(out)):
        abundarr = np.array([abund['C'][i],abund['O'][i]])
        
        if np.isnan(abundarr).any():
            c2ostr = '$nan_{nan}^{+nan}$'
        else:
            for el in elements:
                cent = logeps[el][i]
                err = [errlo[el][i],errhi[el][i]]
                abndone[el] = unumpy.uarray(([cent,cent],err))

            c2o = 10**(abndone['C']-abndone['O'])
            c2ostr = '$%.2f_{-%.2f}^{+%.2f}$' % (c2o[0].nominal_value,c2o[0].std_dev(),c2o[1].std_dev() )
            c2oarr = np.append(c2oarr,c2o[0].nominal_value)

#name,vmag,d,teff,logg,monh,ni,no,o,olo,ohi,nc,c,clo,c
        a = '\\\[1.5ex] %s & %.2f & %d & %d & %.2f & %s & %.2f & %.2f & %d & $%.2f_{%.2f}^{+%.2f}$ & %d & $%.2f_{%.2f}^{+%.2f}$ & %s \\\ \n ' % (out['name'][i],out['vmag'][i],out['d'][i],out['teff'][i],out['logg'][i],out['pop_flag'][i],out['monh'][i],out['ni_abund'][i],out['o_nfits'][i],abund['O'][i],out['o_staterrlo'][i],out['o_staterrhi'][i],out['c_nfits'][i],abund['C'][i],out['c_staterrlo'][i],out['c_staterrhi'][i],c2ostr)
        a = a.replace('$nan_{nan}^{+nan}$',r'\nd')
        a = a.replace('None',r'\nd')
        a = a.replace('nan',r'\nd')

        outstr.append(a)


    c2ohist = np.histogram(c2oarr,bins = [0.,p.coThresh,1000.])[0]
    # Total number of c2o measurments
    texdict['nStarsTot'] = len(out)
    texdict['ncoStarsTot'] = len(c2oarr)
    texdict['ncoGtThresh'],texdict['ncoLtThresh'] = tuple(c2ohist)
    texdict['coMin'],texdict['coMax'] = c2oarr.min(),c2oarr.max()


    if texcmd:
        return texdict

    if save:
        f = open('Thesis/tables/bigtable.tex','w')
        f.writelines(outstr)
