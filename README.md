## Spatial Infinite Relational Model


# Linux build instructions

First make sure your linux machine has boost
`sudo apt-get install libboost-all-dev`

Assume a recent anaconda install, and your root anaconda install is in 
$ANACONDA_ROOT

1. clone repo
2. `cd irm`
3. `mkdir build`
4. `cd build`
5. `cmake -DCMAKE_INSTALL_PREFIX=$ANACONDA_ROOT/ ../`
6. `make -j 8`
7. `make install`


note there is a chance you will have to set LD_LIBRARY_PATH to `$ANACONDA_ROOT/lib`

To validate, try running `tests/modelexample.ipynb` from an ipython notebook
