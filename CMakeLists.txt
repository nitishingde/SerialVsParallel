cmake_minimum_required(VERSION 3.17)

project(SerialVsParallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(Pi
        src/Pi/main.cpp
        src/Pi/Pi.cpp src/Pi/Pi.h
)

target_link_libraries(Pi
    PUBLIC
        OpenMP::OpenMP_CXX
)

add_executable(OpenCLBasics
        src/OpenCLBasics/main.cpp
        src/Utility.cpp src/Utility.h
)

target_link_libraries(OpenCLBasics
    PUBLIC
        OpenCL
)

add_custom_target(CopyScripts ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        ${CMAKE_CURRENT_BINARY_DIR}/
)

add_dependencies(OpenCLBasics CopyScripts)
