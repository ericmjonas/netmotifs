Linux Build notes

sudo apt-get install libboost-all-dev

Assume a recent anaconda install, and your root anaconda install is in 
$ANACONDA_ROOT

1. clone repo
2. `cd irm`
3. `mkdir build`
4. `cd build`
5. `cmake -DCMAKE_INSTA__PRE_FIX=$ANACONDA_ROOT/ ../`
6. `make -j 8`
7. `make install`

note there is a chance you will have to set LD_LIBRARY_PATH to `$ANACONDA_ROOT/lib`


