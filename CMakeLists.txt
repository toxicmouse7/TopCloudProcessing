cmake_minimum_required(VERSION 3.21)
project(top)

set(CMAKE_CXX_STANDARD 20)

find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(top main.cpp CloudProcessor.hpp)
target_link_libraries(top ${PCL_LIBRARIES})
target_link_libraries(top ${OpenCV_LIBS})