cmake_minimum_required(VERSION 3.17)

project(Pi)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(Pi src/main.cpp)

target_sources(Pi
    PRIVATE
        src/Pi.cpp
)

target_include_directories(Pi
    PRIVATE
        src/Pi.h
)

target_link_libraries(Pi
    PUBLIC
        OpenMP::OpenMP_CXX
)
