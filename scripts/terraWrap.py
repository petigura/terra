from argparse import ArgumentParser
import pandas
import terra

parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('args',nargs='+',type=str,help='file[s] function keywords')
args  = parser.parse_args()
line  = int(args.args[-1])
files = args.args[:-1]

dL = [dict( pandas.read_csv(l,index_col=0).ix[line] ) for l in files]
for d in dL:
    exec("terra.%(name)s(d)" % d)


