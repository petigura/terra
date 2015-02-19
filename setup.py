#
# Here's a sample setuptools that builds the cython and fortran extensions
# However, I gave up and just made a simple makefile.
#


from __future__ import division, absolute_import, print_function
from numpy.distutils.core import Extension
import os
from Cython.Distutils import build_ext
import numpy
include_dirs = [numpy.get_include()]
ext_ffa = [
    Extension(
        name="FFA_cy", 
		sources=["FFA/FFA_cy.pyx"],
        include_dirs=include_dirs,
        ),
    Extension(name="FBLS_cy",
    	sources=["FFA/FBLS_cy.pyx"],
    	include_dirs=include_dirs
    	),
    Extension(
    	name="BLS_cy",
        sources=["FFA/BLS_cy.pyx"],
        include_dirs=include_dirs
        ),
    Extension(
    	name="FFA_cext",
    	sources=["FFA/FFA_cext.pyx","FFA/FFA.c"],
        include_dirs=include_dirs
        ),
    Extension(
    	name="FBLS_cext",
        sources=["FFA/FBLS_cext.pyx","FFA/FBLS.c"],
        include_dirs=include_dirs
        ),
    Extension(
    	name="BLS_cext",
        sources=["FFA/BLS_cext.pyx","FFA/BLS.c"],
        include_dirs=include_dirs
        ),
	Extension(
		name="fold",
        sources=["FFA/fold.pyx"],
        include_dirs=include_dirs),
]

ext_transit = [
    Extension(
	name = 'occultsmall',
    sources = ['terra/transit/occultsmall.f']
    )
]	

if __name__ == "__main__":
    # Cython extension
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext

    setup(
        name="terra",
    	ext_modules = ext_ffa,
        cmdclass = {'build_ext': build_ext},
    )

    # Use numpy to run f2py
    from numpy.distutils.core import setup

    setup(
        name="terra",
        author="Erik Petigura",
        author_email="epetigura@berkeley.edu",
        packages=["terra"],
        ext_modules = ext_transit,
        description="Pipeline to search for transiting planets.",
    )

