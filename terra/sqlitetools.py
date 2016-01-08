import os
import sqlite3

def create_table(dbfile, schemafile):
    """
    Checks if sqlite table exists. If not, creates it.
    """
    if not os.path.isfile(dbfile):
        print "creating {}".format(dbfile)
        with open(schemafile) as f:
            schema = f.read()

        con = sqlite3.connect(dbfile)
        with con:
            cur = con.cursor()
            cur.execute(schema)

def insert_dict(d, table, dbfile, schemafile):
    """
    Inserts dictionary into sqlite3 database. Also runs create_table.
    """
    create_table(dbfile, schemafile)
    columns = ', '.join(d.keys())
    placeholders = ':'+', :'.join(d.keys())
    sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table, columns, placeholders)
    print sql
    con = sqlite3.connect(dbfile,60)
    with con:
        cur = con.cursor()
        cur.execute(sql,d)
