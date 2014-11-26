#!/usr/bin/env python 

from flask import Flask, render_template, request, url_for
import sqlite3
import os.path
import pandas as pd
from cStringIO import StringIO as sio
import copy
import k2_catalogs
from time import strftime

cat = k2_catalogs.read_cat()
cat.index = cat.epic.astype(str)

host = os.environ['K2WEBAPP_HOST']
app = Flask(__name__)

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
    templateVars = { 
        "imagepars":imagepars,
                 }
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

    return  outstr

def get_display_vetting_templateVars(starname_url):
    dbpath = os.path.join(tps_basedir0,'scrape.db')
    print "connecting to database %s" % dbpath 

    # Grab the unique id for candidate #
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    query = """
SELECT id from candidate 
GROUP BY starname
HAVING id=MAX(id)
AND starname=%s""" % starname_url
    cur.execute(query)
    id, = cur.fetchone()

    # Capture output from form.
    keys = request.form.keys()
    if len(keys)==0:
        pass
    if (keys.count('is_eKOI')==1) or (keys.count('not_eKOI')==1):
        if keys.count('is_eKOI')==1:
            d = dict(is_eKOI=1)
        if keys.count('not_eKOI')==1:
            d = dict(is_eKOI=0)
        d['is_eKOI_date']=strftime("%Y-%m-%d %H:%M:%S")

        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        sqlcmd = "UPDATE candidate SET is_eKOI=?,is_eKOI_date=? WHERE id=?"
        values = (d['is_eKOI'],d['is_eKOI_date'],id)
        cur.execute(sqlcmd,values)
        con.commit()
        con.close()

    con = sqlite3.connect(dbpath)
    query = "SELECT * from candidate WHERE id=%i" % id
    df = pd.read_sql(query,con)
    con.close()

    starname = starname_url

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
    
    chartkw = dict(
        coords = cat['ra dec'.split()].itertuples(index=False),
        starcoords = cat.ix[[starname]]['ra dec'.split()].itertuples(index=False),
        starname = starname
    )
    
    templateVars = dict(templateVars,**chartkw)
    templateVars['is_eKOI_string'] = is_eKOI_string(dfdict)
    return templateVars


        
@app.route('/vetting/<starname_url>',methods=['GET','POST'])
def display_vetting(starname_url):
    templateVars = get_display_vetting_templateVars(starname_url)
    html = render_template('vetting_template.html',**templateVars)

    return html

@app.route('/vetting/list',methods=['GET','POST'])
def display_vetting_list():

    starname_list = '202092659 202091740 202135853 202087553 202068686 202072485 202126877 202126880 202094117'.split()
    starname_url = starname_list[0]
    templateVars = get_display_vetting_templateVars(starname_url)    
    templateVars['starname_list'] = starname_list
    return render_template('vetting_session_template.html',**templateVars)

if __name__=="__main__":
    app.run(host=host,port=25000,debug=True)
