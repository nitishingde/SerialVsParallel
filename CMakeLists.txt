cmake_minimum_required(VERSION 3.17)

project(SerialVsParallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

add_custom_target(CopyScripts ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources
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

add_executable(Prime
        src/Prime/main.cpp
        src/Prime/Prime.cpp src/Prime/Prime.h
)
target_link_libraries(Prime
    PUBLIC
        Utility
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

add_executable(MatrixMultiplication
        src/MatrixMultiplication/main.cpp
        src/MatrixMultiplication/MatrixMultiplication.cpp src/MatrixMultiplication/MatrixMultiplication.h
)
target_link_libraries(MatrixMultiplication
    PUBLIC
        OpenCL
        OpenMP::OpenMP_CXX
        Utility
)
add_dependencies(MatrixMultiplication CopyScripts)

add_executable(ImageProcessing
        src/ImageProcessing/main.cpp
        src/ImageProcessing/ImageProcessing.cpp src/ImageProcessing/ImageProcessing.h
)
target_link_libraries(ImageProcessing
    PUBLIC
        OpenCL
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        Utility
)
add_dependencies(ImageProcessing CopyScripts)

add_executable(Graph
        src/Graph/main.cpp
        src/Graph/Graph.cpp src/Graph/Graph.h
)
target_link_libraries(Graph
    PUBLIC
        OpenMP::OpenMP_CXX
        Utility
)
add_dependencies(Graph CopyScripts)
