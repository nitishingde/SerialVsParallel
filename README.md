# SerialVsParallel

[![nitishingde - SerialVsParallel](https://img.shields.io/static/v1?label=nitishingde&message=SerialVsParallel&color=blue&logo=github)](https://github.com/nitishingde/SerialVsParallel)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE.md)

- [SerialVsParallel](#serialvsparallel)
  - [Projects](#projects)
  - [Reference materials](#reference-materials)

## Projects
- [Pi calculation](docs/Pi.md)
- [Matrix Multiplication](docs/MatrixMultiplication.md)
- [Image processing](docs/ImageProcessing.md)
- [Graph](docs/Graph.md)

Project|Serial|OpenMP|OpenCL|MPI|OpenMP+MPI
-|-|-|-|-|-
Pi calculation|[Yes](docs/Pi.md#serial-implementation)|[Yes](docs/Pi.md#openmp-implementation)|Yes|[Yes](docs/Pi.md#mpi-implementation)|[Yes](docs/Pi.md#mpi-openmp-hybrid-implementation)
Matrix Multiplication|[Yes](docs/MatrixMultiplication.md#serial-implementation)|[Yes](docs/MatrixMultiplication.md#openmp-implementation)|[Yes](docs/MatrixMultiplication.md#opencl-implementation)|-|-
Image processing: Nearest neighbour interpolation image scaling (nni)|Yes|Yes|Yes|-|-
Graph: Breadth first search (bfs)|Yes|Yes|Yes|-|-
Graph: Single source shortest path (sssp)|Yes|-|Yes|-|-

## Reference materials

- OpenMP
  - [Tim Mattson's yt playlist](https://www.youtube.com/watch?v=nE-xN4Bf8XI&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG)
  - [Lawrence Livermore National Laboratory](https://hpc.llnl.gov/training/tutorials/openmp-tutorial)
  - [Jaka's corner](http://jakascorner.com/blog/)
- OpenCL
  - [Texas Instrumentation docs](http://downloads.ti.com/mctools/esd/docs/opencl/index.html)
  - [Leonardo Araujo Santos blog](https://leonardoaraujosantos.gitbook.io/opencl/)
  - [OpenCL 1.2 reference manual](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/)
- HPC
  - [UC Berkley CS 267](https://people.eecs.berkeley.edu/~demmel/cs267_Spr15/)
  - [UC Berkley CS 267 yt playlist](https://www.youtube.com/playlist?list=PLkFD6_40KJIyX8nEjk6oTLWohdVhjjP3X)
- FAQs
  - [Determine max global work group size based on device memory in OpenCL? ](https://stackoverflow.com/questions/23017005/determine-max-global-work-group-size-based-on-device-memory-in-opencl)
  - [Questions about global and local work size](https://stackoverflow.com/questions/3957125/questions-about-global-and-local-work-size)
