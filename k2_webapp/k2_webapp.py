#!/usr/bin/env python 
from cStringIO import StringIO as sio
import sqlite3
import os.path
import copy
from time import strftime

import numpy as np

from flask import Flask #  creating a flask application,
from flask import render_template # render a HTML template with the given context variables
from flask import request # access the request object which contains the request data
from flask import url_for # get the URL corresponding to a view
from flask import session # store and retrieve session variables in every view
from flask import redirect # redirect to a given URL
from flask import request # access the request object which contains the request data
from flask import flash  # to display messages in the template

import pandas as pd
import k2_catalogs

host = os.environ['K2WEBAPP_HOST']
host = "127.0.0.1"

app = Flask(__name__)
app.secret_key = os.urandom(24)

K2_ARCHIVE = os.environ['K2_ARCHIVE']
K2_ARCHIVE_URL = K2_ARCHIVE.replace(
    "/project/projectdirs/m1669/www/",
    "http://portal.nersc.gov/project/m1669/"
    )

tps_basedir0 = '/project/projectdirs/m1669/www/K2/TPS/C0_11-12/'
phot_basedir0 = '/project/projectdirs/m1669/www/K2/photometry/C0_11-12/'

phot_plots = """\
width ext
33% .ff-frame.png 
50% .ff.png 
33% _xy.png 
50% _gp_time_xy.png
"""
phot_plots = pd.read_table(sio(phot_plots),sep='\s')
def phot_imagepars(starname):
    """
    Return image parameters given the starname
    """
    phot_basedir = copy.copy(phot_basedir0)
    phot_basedir = phot_basedir.replace(
        "/project/projectdirs/m1669/www/",
        "http://portal.nersc.gov/project/m1669/")                         
    phot_basename = os.path.join(phot_basedir,'output',starname)
    phot_plots['url'] = phot_basename + phot_plots['ext']
    imagepars = list(phot_plots['url width'.split()].itertuples(index=False))
    return imagepars

@app.route('/photometry/<starname>')
def display_photometry(starname):
    imagepars = phot_imagepars(starname)
    templateVars = {"imagepars":imagepars}
    return render_template('photometry_template.html',**templateVars)

tps_plots = """\
width ext
100% .grid.pk.png
50% .grid.lc.png
"""
tps_plots = pd.read_table(sio(tps_plots),sep='\s')

def tps_imagepars(starname):
    """
    Return image parameters given the starname
    """
    tps_basedir = copy.copy(tps_basedir0)
    tps_basedir = tps_basedir.replace(
        "/project/projectdirs/m1669/www/",
        "http://portal.nersc.gov/project/m1669/")                         
    tps_basedir = os.path.join(tps_basedir,'output/%s/%s' % (starname,starname))
    tps_plots['url'] = tps_basedir + tps_plots['ext']
    imagepars = list(tps_plots['url width'.split()].itertuples(index=False))
    return imagepars

def is_eKOI_string(d):
    """
    Return a string explaining the disposition status of eKOI

    Parameters
    -----------
    d : dictionary with 
        - is_eKOI
        - is_eKOI_date
    """

    if d['is_eKOI']==None:
        outstr = "No disposition" % d
    else:
        if d['is_eKOI']==1:
            outstr = "Designated as eKOI on %(is_eKOI_date)s " % d
        if d['is_eKOI']==0:
            outstr = "Designated as not eKOI on %(is_eKOI_date)s " % d

    return outstr

def is_EB_string(d):
    """
    Return a string explaining the disposition status of EB

    Parameters
    -----------
    d : dictionary with 
        - is_EB
        - is_EB_date
    """

    is_EB = d['is_EB']
    
    if is_EB==None:
        outstr = "No disposition" % d
    else:
        outstr = "Designated is %s on %s " % (is_EB,d['is_EB_date'])
    return outstr


def is_eKOI_insert(dbpath,dbidx):
    # Capture output from form.
    keys = request.form.keys()
    if len(keys)==0:
        pass
    if (keys.count('is_eKOI')==1) or (keys.count('not_eKOI')==1):
        if keys.count('is_eKOI')==1:
            d = dict(is_eKOI=1)
        if keys.count('not_eKOI')==1:
            d = dict(is_eKOI=0)
        d['is_eKOI_date'] = strftime("%Y-%m-%d %H:%M:%S")

        sqlcmd = "UPDATE candidate SET is_eKOI=?,is_eKOI_date=? WHERE id=?"
        values = (d['is_eKOI'],d['is_eKOI_date'],dbidx)

#        dbpath = 
        con = sqlite3.connect(dbpath)
        with con:
            cur = con.cursor()
            cur.execute(sqlcmd,values)

def is_EB_insert(dbpath,dbidx):
    # Capture output from form.
    keys = request.form.keys()
    print keys
        
    if np.sum([keys.count(k) for k in is_EB_buttons.keys()])==1:
        values = (keys[0],strftime("%Y-%m-%d %H:%M:%S"),dbidx)
        sqlcmd = "UPDATE candidate SET is_EB=?,is_EB_date=? WHERE id=?"
        con = sqlite3.connect(dbpath)
        with con:
            cur = con.cursor()
            cur.execute(sqlcmd,values)
    else:
        pass

is_EB_buttons = {
    'Y_SE':'Y Secondary Eclipse',
    'Y_OOT':'Y OOT Variability',
    'N':'N'
}

def get_tpspath(run,starname_url):
    tpspath = os.path.join(K2_ARCHIVE,'TPS/%s/' % run )
    return tpspath

def starname_to_dbidx(dbpath,starname):
    print "connecting to database %s" % dbpath 

    con = sqlite3.connect(dbpath)
    with con:
        cur = con.cursor()
        query = """
        SELECT id from candidate 
        GROUP BY starname
        HAVING id=MAX(id)
        AND starname=%s""" % starname

    cur.execute(query)
    dbidx, = cur.fetchone()
    return dbidx


class Vetter(object):
    def __init__(self,k2_camp,run,starname_url):
        self.k2_camp = k2_camp
        self.run = run
        self.starname_url = starname_url
        self.tpspath = get_tpspath(run,starname_url)
        self.dbpath = os.path.join(K2_ARCHIVE,self.tpspath,'scrape.db')
        cat = k2_catalogs.read_cat(k2_camp)
        cat.index = cat.epic.astype(str)
        self.cat = cat

    def starname_to_dbidx(self):
        return starname_to_dbidx(self.dbpath,self.starname_url)
    def get_display_vetting_templateVars(self):
        cat = self.cat
        dbidx = self.starname_to_dbidx()

        starname = self.starname_url
        run = self.run
        dbpath = self.dbpath
        is_eKOI_insert(dbpath,dbidx)
        is_EB_insert(dbpath,dbidx)

        con = sqlite3.connect(self.dbpath)
        query = "SELECT * from candidate WHERE id=%i" % dbidx
        df = pd.read_sql(query,con)
        con.close()

        if len(df)==0:
            return "Star %s not in %s" % (starname,tps_basedir0)
        if len(df)>1:
            return "Row returned must be unique"

        dfdict = dict(df.iloc[0] )
        table = df['P t0 tdur s2n grass num_trans'.split()]
        tablelong = df
        table,tablelong = map(lambda x : dict(x.iloc[0]),[table,tablelong])

        table['Depth [ppt]'] = 1e3*tablelong['mean']
        templateVars = { 
            "tps_imagepars":tps_imagepars(starname),
            "phot_imagepars":phot_imagepars(starname),
            "table":table,
            "tablelong":tablelong,
            "cattable":cat.ix[starname]
       }

        coords = cat['ra dec'.split()].itertuples(index=False)
        coords = map(list,coords)
        target = dict(cat.ix[starname]['ra dec'.split()])
        target['starname'] = starname
        templateVars['target'] = target

        chartkw = dict(
            coords = coords,
            starcoords = cat.ix[[starname]]['ra dec'.split()].itertuples(index=False),
            starname = starname
        )

        templateVars = dict(templateVars,**chartkw)
        templateVars['is_eKOI_string'] = is_eKOI_string(dfdict)
        templateVars['is_EB_string'] = is_EB_string(dfdict)
        templateVars['is_EB_buttons'] = is_EB_buttons

        templateVars['run'] = run

        templateVars['phot_outdir'] = os.path.join(
            K2_ARCHIVE_URL,'photometry/%s/output/%s/' % (run,starname)
            )
        templateVars['tps_outdir'] = os.path.join(
            K2_ARCHIVE_URL,'TPS/%s/output/%s/' % (run,starname)
            )
        return templateVars

@app.route('/vetting/<k2_camp>/<run>/<starname_url>',methods=['GET','POST'])
def display_vetting(k2_camp,run,starname_url):
    vetter = Vetter(k2_camp,run,starname_url)
    templateVars = vetter.get_display_vetting_templateVars()
    print templateVars['phot_outdir']
    html = render_template('vetting_template_C1.html',**templateVars)
    return html

@app.route('/vetting/list/<k2_camp>/<run>/',methods=['GET','POST'])
def display_vetting_list(k2_camp,run):
    # Handle button input
    if request.method == "POST":
        keys = request.form.keys()
        if keys.count('starname_list')==1:
            starname_list = request.form.get('starname_list', '').split()
            session['starname_list'] = map(str,starname_list)
            session['nstars'] = len(starname_list)
        if keys.count('prev')==1:
            session["starlist_index"]-=1
        if keys.count('next')==1:
            session["starlist_index"]+=1
        if keys.count('clear')==1:
            session.clear()

    dbpath = os.path.join(K2_ARCHIVE,'TPS/%s/scrape.db' % run)

    # Default behavior when the page is first loaded
    if "starname_list" not in session:
        return render_template('vetting_session_start_template.html')    
    if len(session["starname_list"])==0:
        return render_template('vetting_session_start_template.html')    
    if "starlist_index" not in session:
        session["starlist_index"] = 0

    if session['starlist_index'] < 0:
        session['starlist_index'] = 0 
    if session['starlist_index'] >= session['nstars']:
        session['starlist_index'] = session['nstars']-1

    res = query_starname_list(dbpath,session['starname_list'])
    starname_current = res.iloc[ session['starlist_index']]['starname']
    vetter = Vetter(k2_camp,run,starname_current)

    res['starname_current'] = (res['starname']==starname_current)
    res = res.to_dict('records')
    
    templateVars = vetter.get_display_vetting_templateVars()    
    templateVars['res'] = res
    print templateVars['tps_outdir']
    template = render_template('vetting_session_template.html',**templateVars)
    return template

def query_starname_list(dbpath,starname_list):
    con = sqlite3.connect(dbpath)
    with con:
        cur = con.cursor()
        query = """
SELECT starname,is_eKOI,is_EB from candidate 
GROUP BY starname
HAVING id=MAX(id)
AND starname in %s""" % str(tuple(starname_list))
        cur.execute(query)
        res = cur.fetchall()
    
    res = pd.DataFrame(res,columns=['starname','is_eKOI','is_EB'])
    res.index = res.starname
    res = res.ix[starname_list]
    res['is_eKOI_color'] = res.is_eKOI.apply(is_eKOI_to_color)
    res['is_EB_color'] = res.is_EB.apply(is_EB_to_color)
    return res

def is_EB_to_color(s):
    if s==None:
        return 'LightGray'
    elif s[0]=='Y':
        return 'Tomato'
    elif s[0]=='N':
        return 'RoyalBlue'

def is_eKOI_to_color(is_eKOI):
    if is_eKOI==1:
        return 'RoyalBlue'
    elif is_eKOI==0:
        return 'Tomato'
    else:
        return 'LightGray'

if __name__=="__main__":
    app.run(host='0.0.0.0',port=25001,debug=True)
#    app.run(host=host,port=25001,debug=True)
