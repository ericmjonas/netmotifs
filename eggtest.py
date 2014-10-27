import sys

sys.path.append( "src/build/test.egg")
import pycppdeploy
print pycppdeploy.pycpptest.helloworld()
