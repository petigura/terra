from flask import Flask 
from flask import render_template
import psycopg2
import sqlite3
import os.path
import pandas as pd
app = Flask(__name__)
dbpath = os.environ['K2WEBAPP_DB']

#con = psycopg2.connect(host='your_hostname', user='your username', password='your_password', database='your dbname')

@app.route('/')
def hello():
    # Specify any input variables to the template as a dictionary.
    templateVars = { "title" : "Kepler Data Validation",
                     "description" : "A simple inquiry of function." }
    return render_template('jinja_template.html',**templateVars)

@app.route('/epic/<starname>')
def display_plots(starname):
    # Old code for postgresql
    # pgcon = psycopg2.connect(host='scidb2.nersc.gov', user='kp2_admin', 
    # password='H6bY6tME', database='kp2')
    # cursor = pgcon.cursor()

    # con = sqlite3.connect('/project/projectdirs/m1669/www/K2/TPS/C0_10-10/scrape.db')
    print "connecting to database %s" % dbpath 
    con = sqlite3.connect(dbpath)
    cursor = con.cursor()

    query = "select grid_basedir from candidate where starname=?" 
    cursor.execute( query, (starname,) )
    (phot_basedir,) = cursor.fetchone()


    df = pd.read_sql("select * from candidate where starname=%s" % starname,con)
    phot_basedir = phot_basedir.replace("/project/projectdirs/m1669/www/",
                                        "http://portal.nersc.gov/project/m1669/")                         
    phot_plot_filename = "%s.grid.pk.png" % starname
    phot_url = os.path.join(phot_basedir,phot_plot_filename)

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

if __name__=="__main__":
#    app.run(host="0.0.0.0",port=25000,debug=True)
    app.run(host="localhost",port=25000,debug=True)
