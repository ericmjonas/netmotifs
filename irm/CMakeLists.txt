CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(irm)

set( CMAKE_CXX_FLAGS "-Wall -m64 -O2 -std=c++11 " )

# boost requirements

find_package(Boost 1.53.0 COMPONENTS  python system  REQUIRED)

# notes: 
# http://stackoverflow.com/questions/1937232/linking-to-a-dynamic-library-on-a-mac-with-full-path
# we tried : http://stackoverflow.com/questions/7233383/cmake-on-mac-os-x-link-libraries-with-fullpath but with no real success


include_directories(${Boost_INCLUDE_DIRS})

set(PYTHON_LIBRARY "/Users/jonas/anaconda/lib/libpython2.7.dylib")
set(PYTHON_INCLUDE "/Users/jonas/anaconda/include/python2.7")

include(FindPythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

#include_directories(${PYTHON_INCLUDE})


add_executable(hello 
   helloworld.cc
                     
)

add_library(irm SHARED
   relation.cc

)

add_library(pyirm SHARED
    pyirm.cc
)



TARGET_LINK_LIBRARIES( hello ${Boost_LIBRARIES} )

TARGET_LINK_LIBRARIES( irm ${Boost_LIBRARIES} )

TARGET_LINK_LIBRARIES( pyirm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

INSTALL(TARGETS irm LIBRARY DESTINATION lib)

set_target_properties(pyirm PROPERTIES PREFIX "" SUFFIX ".so")

set(pyirm_install_DIR lib/site-python/pyirm)

install(TARGETS pyirm LIBRARY DESTINATION ${pyirm_install_DIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py 
              ${CMAKE_CURRENT_SOURCE_DIR}/util.py
              DESTINATION ${pyirm_install_DIR}
)
