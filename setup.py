from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension(name="FFA_cy",
                         sources=["FFA_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="FFA_cext",
                         sources=["FFA_cext.pyx","FFA.c"],
                         include_dirs=[numpy.get_include()])
                         ]

setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
