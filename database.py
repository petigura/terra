import sqlite3
import numpy as np
import matplotlib.mlab as mlab
import os
import matplotlib.pyplot as plt
import readstars

class Database:
    def __init__(self):
        """
        Initialize class and define variables that variables common to
        multiple methods.
        """
        self.dbfile = ['stars.db']
        self.conn = sqlite3.connect(self.dbfile[0])
        self.cur  = self.conn.cursor()

    def cleandb(self):
        """
        Spawns the `rm` command to clean up old databases.
        """
        for file in self.dbfile:
            if os.path.exists(file):
                os.system('rm '+ file)
        pass
            
    def addstars(self):
        """
        Insert data from the stars structure into database.
        """
        idxarr,oidarr = matchstars.res2id('smefiles/myresults.sim')
        stars = readstars.ReadStars('keck-fit-lite.sav')
        nstars = len(stars.name)
        fields = ['name','vsini','teff',
                  'o_abund','o_staterr','o_nierr',
                  'c_abund','c_staterr']


        ### CREATE COMMAND ###
        createcmd = 'CREATE TABLE stars ('
        sqlcol    = '('  #List of columns in the sql table
        data      = '('
        fmt       = ''

        for tag in fields:
            exec('field = stars.'+tag)
            t = type(field[0])
            if t is np.float32:
                createcmd = createcmd+tag+' REAL,'
                sqlcol = sqlcol+tag+','
            elif t is np.str:
                createcmd = createcmd+tag+' TEXT,'
                sqlcol = sqlcol+tag+','
            elif t is np.ndarray:
                sqlcol = sqlcol+tag+'lo,'+tag+'hi,'
                createcmd = createcmd+tag+'lo REAL,'+tag+'hi REAL,'

        sqlcol = sqlcol+'oid)'
        createcmd = createcmd+'oid INT)'
        data = data+' "" )'
        fmt = fmt+'%s'

#        self.cur.execute(createcmd)
        for i in np.arange(nstars):
            datlist = []
            for tag in fields:
                exec('t = stars.'+tag+'[i]')
                if type(t) is np.ndarray:
                    datlist.append(t[0])
                    datlist.append(t[1])                    
                else:
                    datlist.append(t)
            datlist.append('')
            datlist = str(datlist)
            datlist = datlist.replace('[','(')
            datlist = datlist.replace(']',')')


            insertcmd = 'INSERT INTO stars '+sqlcol+'VALUES '+datlist
            self.cur.execute(insertcmd)


        self.conn.commit()

    def junk():
        #load the data from csv into database.
        ################ NAMES ########################

        self.cur.execute('CREATE TABLE names (state TEXT, dem TEXT, gop '+
                         'TEXT, ind TEXT, incum TEXT)')
        names  = mlab.csv2rec('candidate_names.txt')
        for line in names:
            self.cur.execute('INSERT INTO names (state,dem,gop,ind,'+
                             'incum) VALUES '+str(line))

        ################# POLL DATA ####################

        self.cur.execute('CREATE TABLE poll (day float, state text, dem int, '+
                         'gop int, ind int)')
        poll  = mlab.csv2rec('senate_polls.csv')
        for line in poll:
            line = (line['day'],line['state'],line['dem']
                    ,line['gop'],line['ind'])
            self.cur.execute('INSERT INTO poll (day,state,dem,gop,ind)'+
                              ' VALUES '+str(line))
        ################## STATE ABBREVIATIONS  ########        

        self.cur.execute('CREATE TABLE abbr (full TEXT, code TEXT)')
        abbr  = mlab.csv2rec('stateabbr.txt',delimiter='\t')
        for line in abbr:
            self.cur.execute('INSERT INTO abbr (full,code) VALUES '+ str(line))

        self.conn.commit()
    
