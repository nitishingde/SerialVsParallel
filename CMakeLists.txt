cmake_minimum_required(VERSION 3.17)

project(SerialVsParallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)

add_custom_target(CopyScripts ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        ${CMAKE_CURRENT_BINARY_DIR}/
)

add_library(Utility
    src/Utility.cpp src/Utility.h
)
target_link_libraries(Utility
    PUBLIC
        OpenCL
        MPI::MPI_CXX
)

add_executable(Pi
        src/Pi/main.cpp
        src/Pi/Pi.cpp src/Pi/Pi.h
)
target_link_libraries(Pi
    PUBLIC
        OpenMP::OpenMP_CXX
        OpenCL
        Utility
        MPI::MPI_CXX
)
add_dependencies(Pi CopyScripts)

add_executable(MPI_Pi
        src/Pi/mpi_main.cpp
        src/Pi/Pi.cpp src/Pi/Pi.h
)
target_link_libraries(MPI_Pi
    PUBLIC
        OpenMP::OpenMP_CXX
        OpenCL
        Utility
        MPI::MPI_CXX
)

add_executable(OpenCLBasics
        src/OpenCLBasics/main.cpp
)
target_link_libraries(OpenCLBasics
    PUBLIC
        OpenCL
        Utility
)
add_dependencies(OpenCLBasics CopyScripts)
