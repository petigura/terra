### Module for dealing with environment variables
import os
import sys

def envset(envlist):
    """
    Checks a list of environment variables and makes sure that they are set.
    If not exit.
    """
    envfail = [] # bucket for failed environment variables
    for env in envlist:
        if not os.environ.has_key(env):
            envfail.append(env)

    if len(envfail) == 0:
        pass
    else:
        print '-'*50
        for env in envfail:
            print "Set the following enviroment variable: %s" % env
        print '-'*50            
        sys.exit(1)
