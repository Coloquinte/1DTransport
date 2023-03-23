
cmake_minimum_required(VERSION 3.10)
PROJECT(TRANSPORTATION_1D)
SET(CMAKE_CXX_STANDARD 17)


enable_testing()
#FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
#add_executable(test_transportation_1d cpp/transportation_1d.cpp cpp/test_transportation_1d.cpp)
#add_test(test_transportation_1d test_transportation_1d)

add_executable(run_transportation_1d cpp/transportation_1d.cpp cpp/main.cpp)
