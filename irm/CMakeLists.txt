CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(irm)

set( CMAKE_CXX_FLAGS "-Wall -m64 -O2 -std=c++11 " )

# boost requirements

find_package(Boost 1.40.0 COMPONENTS thread unit_test_framework program_options filesystem python system REQUIRED)

add_executable(hello 
   helloworld.cc
                     
)

add_library(irm SHARED
   relation.cc

)


TARGET_LINK_LIBRARIES( hello ${Boost_LIBRARIES} )

TARGET_LINK_LIBRARIES( irm ${Boost_LIBRARIES} )
