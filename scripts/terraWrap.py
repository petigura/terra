from argparse import ArgumentParser
import pandas
import terra
import matplotlib
matplotlib.use('Agg')

parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('args',nargs='+',type=str,help='file[s] function keywords')
parser.add_argument('--multi',type=int,default=1,help='')
args  = parser.parse_args()

index = args.args[-1]
files = args.args[:-1]
multi = args.multi

def func(f):
    df = pandas.read_csv(f,index_col=0)
    if index[0]=='0':
        df.index = df.outfile.apply(lambda x : x.split('/')[-1][:-3])

    return dict(df.ix[index]) 

dL = map(func,files)

def last2(s):
    sL = s.split('/')[-2:]
    sL = tuple(sL)
    return '%s/%s' % sL 

if multi > 1:
    outfile0    = dL[0]['outfile']
    if multi > 2:
        outfile = outfile0.replace('grid','grid%i'  % (multi-1))
    else:
        outfile = outfile0

    newoutfile = outfile0.replace('grid','grid%i'  % multi)


    print "copying %s to %s" %  tuple( map(last2,[outfile,newoutfile]) )
    terra.multiCopyCut( outfile , newoutfile )
    for d in dL:
        d['outfile'] = newoutfile

for d in dL:
    exec("terra.%(name)s(d)" % d)


