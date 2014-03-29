CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(irm)

set( CMAKE_CXX_FLAGS "-Wall -m64 -g -O3 -std=c++11 -lpthread" )

# boost requirements
if(APPLE)
set(Boost_USE_STATIC_LIBS   ON)
else(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif(APPLE)

find_package(Boost 1.48.0 COMPONENTS  python system  REQUIRED)

# notes: 
# http://stackoverflow.com/questions/1937232/linking-to-a-dynamic-library-on-a-mac-with-full-path
# we tried : http://stackoverflow.com/questions/7233383/cmake-on-mac-os-x-link-libraries-with-fullpath but with no real success


include_directories(${Boost_INCLUDE_DIRS})

if(APPLE)
set(PYTHON_LIBRARY "/Users/jonas/anaconda/lib/libpython2.7.dylib")
set(PYTHON_INCLUDE "/Users/jonas/anaconda/include/python2.7")
endif(APPLE)

include(FindPythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

add_executable(benchmark
   benchmark.cc
)

add_executable(futuretest
   futuretest.cc
)

add_executable(disttest
        disttest.cc
)

add_library(irm 
   relation.cc
   componentmodels.cc
   componentslice.cc
   componentmh.cc
   slicesample.cc

)

add_library(pyirm SHARED
    pyirm.cc
    pyirm_helpers.cc
)



TARGET_LINK_LIBRARIES( benchmark ${Boost_LIBRARIES} irm ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES( futuretest ${Boost_LIBRARIES} irm ${PYTHON_LIBRARIES})

TARGET_LINK_LIBRARIES( pyirm irm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

set_target_properties(pyirm PROPERTIES PREFIX "" SUFFIX ".so")

execute_process(COMMAND python -c "from distutils import sysconfig;print(sysconfig.get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

set(pyirm_install_DIR "${PYTHON_SITE_PACKAGES}/irm")
message(STATUS "Installing python to ${PYTHON_SITE_PACKAGES}")

install(TARGETS pyirm LIBRARY DESTINATION ${pyirm_install_DIR})

FILE(GLOB py_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

install(FILES ${py_files}
              DESTINATION ${pyirm_install_DIR})

FILE(GLOB py_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/data/*.py")

install(FILES ${py_files}
              DESTINATION ${pyirm_install_DIR}/data)

FILE(GLOB py_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plots/*.py"     "${CMAKE_CURRENT_SOURCE_DIR}/plots/*.template")

install(FILES ${py_files}
              DESTINATION ${pyirm_install_DIR}/plots)
