import argparse
import tfind

desc = """
Wrapper around tfind
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('grid', type=str, help='grid file *.h5')
parser.add_argument('cmds',nargs='+',type=str,help='dt, cal, mqcal')
parser.add_argument('--lc', type=str,help='grid requires valid lc file')

parser.add_argument('--debug', action="store_true", default=None,help='Turn A off',)



args = parser.parse_args()
if args.lc != None:
    grid = tfind.Grid(args.grid,args.lc)
else:
    grid = tfind.Grid(args.grid)

if args.debug:
    print "shorter run"
    grid.P2 = 500

for cmd in args.cmds:
    eval_str = "grid.%s()" % (cmd)
    eval(eval_str)
    print eval_str
