CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if(CMAKE_CONFIGURATION_TYPES)
   list(APPEND CMAKE_CONFIGURATION_TYPES EGG)
   list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
   set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
     "Add the configurations that we need"
     FORCE)
 endif()


PROJECT(irm)
set(PACKAGE_NAME irm)

set( CMAKE_CXX_FLAGS "-Wall -m64 -g -O3 -std=c++11 -lpthread" )

# boost requirements
if(APPLE)
set(Boost_USE_STATIC_LIBS   ON)
else(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif(APPLE)

find_package(Boost 1.48.0 COMPONENTS  python system thread  REQUIRED)

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
   parrelation.cc
   componentmodels.cc
   componentslice.cc
   componentmh.cc
   slicesample.cc

)

add_library(pyirm SHARED
    pyirm.cc
    pyirm_helpers.cc
)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

TARGET_LINK_LIBRARIES(irm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(benchmark irm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(futuretest ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(disttest ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

TARGET_LINK_LIBRARIES( pyirm irm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

set_target_properties(pyirm PROPERTIES PREFIX "" SUFFIX ".so")

execute_process(COMMAND python -c "from distutils import sysconfig;print(sysconfig.get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)


IF(CMAKE_BUILD_TYPE MATCHES EGG)
    set(package_install_DIR "${CMAKE_BINARY_DIR}/egg/${PACKAGE_NAME}")
ELSE(CMAKE_BUILD_TYPE MATCHES EGG)
    set(package_install_DIR "${PYTHON_SITE_PACKAGES}/${PACKAGE_NAME}")
ENDIF(CMAKE_BUILD_TYPE MATCHES EGG)



message(STATUS "Installing python package to ${PYTHON_SITE_PACKAGES}")

install(TARGETS pyirm LIBRARY DESTINATION ${package_install_DIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
                  DESTINATION ${package_install_DIR}
                  FILES_MATCHING PATTERN "*.py" PATTERN "build" EXCLUDE)

####### NOW THE EGG -- only for linux? 
###
###

IF(CMAKE_BUILD_TYPE MATCHES EGG)
   install(CODE "MESSAGE(\"Sample install message.\")")
   install(CODE "execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/../buildegg.py ${CMAKE_SOURCE_DIR}/../EGG-INFO ${CMAKE_BINARY_DIR}/egg ${CMAKE_BINARY_DIR}/test.egg)")
ENDIF(CMAKE_BUILD_TYPE MATCHES EGG)
    
