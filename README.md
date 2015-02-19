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
2. cd into traget directory
   ```
   /global/homes/p/petigura/code_carver/terra
   ```
   
3. Bulid the cython and fortran extensions with make. At NERSC, there was a little difficulty getting the fortran and c libraries to compile. Here's what I had to do:

   ```
   module load gcc
   # This will compile the cython extensions but the -dynamic lookup won't pass to the gfort
   LDFLAGS="-L/global/homes/p/petigura/anaconda/lib" SOFLAGS="-fPIC -shared" make
   # Running it a second time will get the fortran code compiled
   make
   ```

4. Add `</path/to/terra>` to $PYTHONPATH
5. Add `</path/to/terra> /bin/` to $PATH
6. Test that everything is working by running the following test

   ```
   python -c "from terra import terra; terra.test_terra()"
   ```
   
   
