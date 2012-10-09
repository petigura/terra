import argparse
import os
import prepro

desc = """
Wrapper around prepro
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('lc', type=str, help='light curve *.h5')
parser.add_argument('cmds',nargs='+',type=str,help='dt, cal, mqcal')
parser.add_argument('--svd_folder', type=str,help='folder containing SVD matrices')
parser.add_argument('--fits',nargs='+',type=str,help='list of input fits files')
args = parser.parse_args()

lc = prepro.Lightcurve(args.lc)
for cmd in args.cmds:
    if cmd=='cal':
        eval_str = "lc.%s('%s')" % (cmd,args.svd_folder)
    elif cmd=='raw':
        eval_str = "lc.%s(args.fits)" % (cmd)
    else:
        eval_str = "lc.%s()" % cmd

    eval(eval_str)
    print eval_str
