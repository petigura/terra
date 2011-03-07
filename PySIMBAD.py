"""
The SIMBAD database has a unique identifier for each object called the 
`oid.` This identifier is a handy way to match objects from different 
catalogs. Using this module is a three-step process.

(1) - Generate a scripted SIMBAD query with the `names2sim` method.  
      Converts array of names into a SIMBAD query.

(2) - Upload the script to SIMBAD script page.  Download the output as
      a file.
      http://simbad.u-strasbg.fr/simbad/sim-fscript


(3) - Parse the output file with `res2id` method.  The output will be two 
      tuples: the SIMBAD oid matched with the *index* of the star in the
      names array.

Future work:
------------
Figure out how to combine the 3 steps into one by communicating with SIMBAD
through urllib2 

"""

import numpy as np
import re


def names2sim(names,cat=''):
    """
    Creates a SIMBAD scripted query from from an of star names.

    cat - SIMBAD must know what catalog the star is from. If names already
          specify the catalog, we're done.  If not set cat to append the
          catalog to the names file.
          
          HD  - Henry-Draper Catalog
          HIP - Hiparcos Catalog

    >>> names = np.array(['14412','4915'])
    >>> sim = names2sim(names,cat='HD')

    To use in a SIMBAD script, one must write to a file
    --------------File Output--------------
    result oid

    echodata -n x0
    query id  14412

    echodata -n x1
    query id  4915
    ---------------------------------------

    Future work:
    Make unit tests.
    """

    query   = np.array([],dtype='|S50')

    #Add whitespace between catalog and star name
    if cat is not '':
        cat += ' '

    #First line of query - specifies we want oid output
    query = np.append(query,'result oid\n')

    for i in np.arange( len(names) ):
        name = names[i]
        name = cat+name

        # Two lines of SIMBAD script per star.  The 'x' is a necessary
        # placehold for subsequent parsing
        queryline = 'echodata -n x%i\nquery id %s\n' % (i,name)
        query = np.append(query,queryline) 

    return query

def names2plx(names):
    query   = np.array([],dtype='|S50')

    #First line of query - specifies we want oid output
    query = np.append(query,'format object form1 "#1: %PLX(V [E])"\n')

    for i in np.arange( len(names) ):
        name = names[i]

        # Two lines of SIMBAD script per star.  The 'x' is a necessary
        # placehold for subsequent parsing
        queryline = 'echodata -n x%i\nquery id %s\n' % (i,name)
        query = np.append(query,queryline) 

    return query
    


def res2id(file):
    """
    Parses SIMBAD query results into matched pairs of (idx,oid)

    Query Results have this form:
    
    :: Header :::::::
    script snipet
    console
    error messages

    :: data :::::::::

    x0#1: 1356191
    x1#1: 1225912
    x2#1: 759900
    x3#1: 936604

    res2id chops off the header and returns    
    0, 1356191
    1, 1225912
    2, 759900
    3, 936604

    Future Work:
    Create unit test.
    """

    idxarr,oidarr = np.array([]),np.array([])

    f = open(file,'r')
    lines = f.readlines()
    for i in np.arange(len(lines)):
        if re.search('#1',lines[i]) is not None:
            idx,sep,oid = lines[i].partition('#1: ')
            idxarr = np.append(idxarr,int(idx[idx.rfind('x')+1:]))
            oidarr = np.append(oidarr,int(oid[:-1]))

    return idxarr,oidarr

def res2plx(file):

    out = []
    
    f = open(file,'r')
    lines = f.readlines()    
    data = False
    for i in np.arange(len(lines)):
        line = lines[i]
        if line.find('::data') != -1:
            data = True

        if re.search('#1',line) is None:
            pass
        elif line.find('~') != -1:
            pass
        elif data is False:
            pass
        else:
            out.append(parseplx(line))

    out = np.array(out,dtype=[('idx',int),('plx',float),('e_plx',float)])

    return out

def parseplx(line):
    #remove brackets
    line = line.replace('[','')
    line = line.replace(']','')
    
    line = line.replace('#1:',' ')
    line = line.split('  ')
    line[0] = line[0][line[0].rfind('x')+1:]
    return tuple(line)
