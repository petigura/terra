import argparse
import tfind

desc = """
Wrapper around tfind
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('grid', type=str, help='grid file *.h5')
parser.add_argument('cmds',nargs='+',type=str,help='dt, cal, mqcal')
parser.add_argument('--lc', type=str,help='grid requires valid lc file')
args = parser.parse_args()

grid = tfind.Grid(args.grid)
for cmd in args.cmds:
    if cmd=='grid':
        eval_str = "grid.%s('%s')" % (cmd,args.lc)
    elif cmd=='itOutRej':
        eval_str = "grid.%s()" % (cmd)
    eval(eval_str)
    print eval_str
