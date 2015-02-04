#!/usr/bin/env python
import terra
import pandas as pd
import sqlite3
from argparse import ArgumentParser

def pp(args):
    con = sqlite3.connect(args.parfile)
    df = pd.read_sql('select * from pp',con,index_col='id')
    d = dict(df.ix[args.index])
    d['outfile'] = args.outfile
    d['path_phot'] = args.path_phot
    terra.pp(d)

def grid(args):
    con = sqlite3.connect(args.parfile)
    df = pd.read_sql('select * from grid',con,index_col='id')
    d = dict(df.ix[args.index])
    d['outfile'] = args.outfile
    terra.grid(d)

def dv(args):
    con = sqlite3.connect(args.parfile)
    df = pd.read_sql('select * from dv',con,index_col='id')
    d = dict(df.ix[args.index])
    d['outfile'] = args.outfile
    terra.data_validation(d)

if __name__=='__main__':

    p = ArgumentParser(description='Wrapper around functions in terra.py')
    subparsers = p.add_subparsers()

    p_pp = subparsers.add_parser('pp', help='Run the preprocessing module')
    p_pp.add_argument(
        'path_phot',type=str,help='photometry file *.fits | *.h5')

    p_pp.add_argument('outfile',type=str,help='output file <*.grid.h5>')
    p_pp.add_argument('parfile',type=str,help='parameter file <*.sqlite>')
    p_pp.add_argument('index',type=str,help='photometry id')
    p_pp.set_defaults(func=pp)

    p_grid = subparsers.add_parser('grid', help='Run the grid search code')
    p_grid.add_argument('outfile',type=str,help='output file <*.grid.h5>')
    p_grid.add_argument('parfile',type=str,help='parameter file <*.sqlite>')
    p_grid.add_argument('index',type=str,help='photometry id')
    p_grid.set_defaults(func=grid)

    p_grid = subparsers.add_parser('dv', help='Run the data validation module')
    p_grid.add_argument('outfile',type=str,help='output file <*.grid.h5>')
    p_grid.add_argument('parfile',type=str,help='parameter file <*.sqlite>')
    p_grid.add_argument('index',type=str,help='photometry id')
    p_grid.set_defaults(func=dv)

    args = p.parse_args()

    # Import here to save time when just calling help.

    args.func(args)
