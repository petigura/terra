# TERRA #

A suite of codes to find extrasolar planets

## Installation Instructions ##

1. clone the git repo
2. cd /global/homes/p/petigura/code_carver/terra
3. Bulid the cython and fortran extensions with make. At NERSC f2py did not like compiling the fortran code with the default ifort compiler. Make sure gfortran is loaded instead.
4. Add ${TERRA}/terra/ to PYTHONPATH
5. Add ${TERRA}/bin/ to PATH

