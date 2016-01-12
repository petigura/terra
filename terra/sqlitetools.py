import os
import sqlite3


def create_table(dbfile, schemafile, if_exists='pass'):
    """
    Checks if sqlite table exists. If not, creates it.
    """
    database_exists = os.path.isfile(dbfile)
    if database_exists and if_exists=='pass':
        return

    if database_exists and if_exists=='replace':
        os.system('rm {}'.format(dbfile))
        _create_table(dbfile, schemafile)
        return

    if not database_exists:
        _create_table(dbfile, schemafile)
        return 

def _create_table(dbfile, schemafile):
    with open(schemafile) as f:
        schema = f.read()

    con = sqlite3.connect(dbfile)
    with con:
        print "creating {}".format(dbfile)
        cur = con.cursor()
        cur.execute(schema)

def insert_dict(d, table, dbfile):
    """
    Inserts dictionary into sqlite3 database
    """
    columns = ', '.join(d.keys())
    placeholders = ':'+', :'.join(d.keys())
    sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table, columns, placeholders)
    print sql
    con = sqlite3.connect(dbfile,60)
    with con:
        cur = con.cursor()
        cur.execute(sql,d)
