#!/usr/bin/env python 
from flask import Flask, render_template, request, url_for
import psycopg2
import sqlite3
import os.path
import pandas as pd

app = Flask(__name__)
dbpath = os.environ['K2WEBAPP_DB']
dbpath = '/project/projectdirs/m1669/www/K2/TPS/C0_11-12/scrape.db'

host = os.environ['K2WEBAPP_HOST']
print "connecting to database %s" % dbpath 

#con = psycopg2.connect(host='your_hostname', user='your username', password='your_password', database='your dbname')

@app.route('/epic/<starname>')
def display_plots(starname):
    # Old code for postgresql
    # pgcon = psycopg2.connect(host='scidb2.nersc.gov', user='kp2_admin', 
    # password='H6bY6tME', database='kp2')
    # cursor = pgcon.cursor()

    con = sqlite3.connect(dbpath)
    cursor = con.cursor()

    query = "select grid_basedir from candidate where starname=?" 
    cursor.execute( query, (starname,) )
    (phot_basedir,) = cursor.fetchone()

    phot_basedir = phot_basedir.replace('/global','')
    df = pd.read_sql("select * from candidate where starname=%s" % starname,con)
    phot_basedir = phot_basedir.replace("/project/projectdirs/m1669/www/",
                                        "http://portal.nersc.gov/project/m1669/")                         
    phot_plot_filename = "%s.grid.pk.png" % starname
    phot_url = os.path.join(phot_basedir,phot_plot_filename)
    print 
    print phot_url


    table = dict(df['P t0 tdur s2n grass num_trans'.split()].iloc[0])
    tablelong = dict(df.iloc[0])

    table['Depth [ppt]'] = 1e3*tablelong['mean']

    templateVars = { "title" : "Kepler Data Validation",
                     "description" : "A simple inquiry of function.","id":id,
                     "phot_plot_filename":phot_plot_filename,
                     "phot_basedir":phot_basedir,
                     "phot_url":phot_url,
                     "table":table,
                     "tablelong":tablelong}

    # pgcon.close()
    con.close()
    return render_template('jinja_template.html',**templateVars)

extname = ".ff-frame.png .ff.png _xy.png _gp_time_xy.png".split()
width = "33% 50% 33% 50%".split()

phot_basedir = '/project/projectdirs/m1669/www/K2/photometry/C0_11-12/'

@app.route('/photometry/<starname>')
def display_photometry(starname):
    phot_basedir = phot_basedir.replace(
        "/project/projectdirs/m1669/www/",
        "http://portal.nersc.gov/project/m1669/")                         
    phot_basename = os.path.join(phot_basedir,'output',starname)

    templateVars = { 
        "title" : "K2 Photometry Viewer",
        "phot_basename":phot_basename,
        "starname":starname,
        "imagepars":zip(extname,width),
                 }
    return render_template('photometry_template.html',**templateVars)

# Insert decision of real / not real into data base 


@app.route('/hello/',methods=['POST'])
def hello():
    return request.form['isreal']

if __name__=="__main__":
    app.run(host=host,port=25000,debug=True)
