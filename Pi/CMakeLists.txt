cmake_minimum_required(VERSION 3.17)

project(Pi)

set(CMAKE_CXX_STANDARD 20)

add_executable(Pi src/main.cpp)

target_sources(Pi
    PRIVATE
        src/Pi.cpp
)

target_include_directories(Pi
    PRIVATE
        src/Pi.h
)
