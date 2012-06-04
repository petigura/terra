import argparse
import h5plus

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , 
                    help='input files passed to glob')

parser.add_argument('out',  type=str   , 
                    help='output h5 file')

parser.add_argument('--diff',nargs='+',type=str,default=[],
                    help='list of fields to be stored individually')



# Unpack arguments
args  = parser.parse_args()
inp   = args.inp
out   = args.out
diff  = args.diff

h5plus.atpy2h5(inp,out,diff=diff)

