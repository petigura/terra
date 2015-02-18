# TERRA #

A suite of codes to find extrasolar planets

## Dependencies ##

### Public Python modules ###
```
- matplotlib / pylab (tested with v1.1.0, 1.3.1)
- NumPy (tested with v1.6.2, 1.8.1)
- SciPy (tested with 0.7.0, 0.10.1, 0.14.0)
- Pandas (v0.14.1)
- emcee
```

## Installation Instructions ##

1. clone the git repo. Will create a `</path/to/terra>` directory
2. cd /global/homes/p/petigura/code_carver/terra
3. Bulid the cython and fortran extensions with make. At NERSC f2py did not like compiling the fortran code with the default ifort compiler. Make sure gfortran is loaded instead.
4. Add `</path/to/terra>` to $PYTHONPATH
5. Add `</path/to/terra> /bin/` to $PATH
6. Test that everything is working by running the following test

   ```
   python -c "from terra import terra; terra.test_terra()"
   ```
   
   
