from argparse import ArgumentParser
import h5plus
import glob

parser = ArgumentParser(description='Inject transit into template light curve.')

parser.add_argument('--files',nargs='+',type=str , help='input file')
parser.add_argument('--out', type=str , help='output h5 file')
parser.add_argument('--diff',nargs='+',type=str,default='all',
                    help='list of fields to be stored individually')



# Unpack arguments
args  = parser.parse_args()
files   = args.files
out   = args.out
diff  = args.diff

h5plus.atpy2h5(files,out,diff=diff)

