cmake_minimum_required(VERSION 3.17)

project(SerialVsParallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(Pi
        src/Pi/main.cpp
        src/Pi/Pi.cpp src/Pi/Pi.h
)

target_link_libraries(Pi
    PUBLIC
        OpenMP::OpenMP_CXX
)