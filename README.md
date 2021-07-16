# SerialVsParallel

[![nitishingde - SerialVsParallel](https://img.shields.io/static/v1?label=nitishingde&message=SerialVsParallel&color=blue&logo=github)](https://github.com/nitishingde/SerialVsParallel)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE.md)

- [SerialVsParallel](#serialvsparallel)
  - [About](#about)
  - [Brief Overview](#brief-overview)
    - [OpenMP](#openmp)
    - [MPI](#mpi)
    - [OpenCL](#opencl)
  - [Projects](#projects)
  - [How to build and run the projects?](#how-to-build-and-run-the-projects)
  - [Reference materials](#reference-materials)

## About

> TL;RD: Repository of examples of parallel algorithms

The objective of this project is to help write parallel algorithms using OpenMP, MPI and OpenCL.
A complete knowledge of these libraries can be found online, and hence won't be discussed here. You can find the resources linked at the [bottom of this page](#reference-materials).

Here we will look more on use case side of things. This is best achieved by leading through examples. First a serial code for the algorithm will be provided, and then we will modify the serial code accordingly to get a parallelized version using a combination of OpenMP, MPI and OpenCL.

Please note that the code for the parallel algorithms isn't guaranteed to be the best/most efficient. Also, most of the programming language related optimizations are avoided, since that is not the objective here (they are often times ugly and impairs readability). For the sake of clarity and better understanding, the code is tried to be kept as simple as possible.

## Brief Overview

### OpenMP

- Open Multi-Processing
- An Application Program Interface (API) that may be used to explicitly direct multi-threaded, shared memory parallelism.
- Comprised of three primary API components:
  - Compiler Directives
  - Runtime Library Routines
  - Environment Variables
- OpenMP Is Not:
  - Meant for distributed memory parallel systems (by itself)
  - Necessarily implemented identically by all vendors
  - Guaranteed to make the most efficient use of shared memory
  - Required to check for data dependencies, data conflicts, race conditions, deadlocks, or code sequences that cause a program to be classified as non-conforming
  - Designed to handle parallel I/O.
  - The programmer is responsible for synchronizing input and output.

### MPI

- Message Passing Interface
- MPI is a specification for the developers and users of message passing libraries. By itself, it is NOT a library - but rather the specification of what such a library should be.
- MPI primarily addresses the message-passing parallel programming model: data is moved from the address space of one process to that of another process through cooperative operations on each process.
- Simply stated, the goal of the Message Passing Interface is to provide a widely used standard for writing message passing programs. The interface attempts to be:
  - Practical
  - Portable
  - Efficient
  - Flexible

### OpenCL

- Open Computing Language
- It is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators.
- SIMD processing.


## Projects
- [Pi calculation](docs/Pi.md)
- [Prime](docs/Prime.md)
- [Matrix Multiplication](docs/MatrixMultiplication.md)
- [Image processing](docs/ImageProcessing.md)
- [Graph](docs/Graph.md)

Project|Serial|OpenMP|MPI|MPI+OpenMP|OpenCL
-|-|-|-|-|-
Pi calculation|[Yes](docs/Pi.md#serial-implementation)|[Yes](docs/Pi.md#openmp-implementation)|[Yes](docs/Pi.md#mpi-implementation)|[Yes](docs/Pi.md#mpi-openmp-hybrid-implementation)|[Yes](docs/Pi.md#opencl-implementation)
Prime: Get the largest prime number and count(prime) below a given upper limit.|Yes|Yes|Yes|Yes|-
Matrix Multiplication|[Yes](docs/MatrixMultiplication.md#serial-implementation)|[Yes](docs/MatrixMultiplication.md#openmp-implementation)|-|-|[Yes](docs/MatrixMultiplication.md#opencl-implementation)
Image processing: Nearest neighbour interpolation image scaling (nni)|[Yes](docs/ImageProcessing.md#serial-implementation)|[Yes](docs/ImageProcessing.md#openmp-implementation)|-|-|[Yes](docs/ImageProcessing.md#opencl-implementation)
Graph: Breadth first search (bfs)|Yes|Yes|-|-|Yes
Graph: Single source shortest path (sssp)|Yes|-|-|-|Yes

## How to build and run the projects?

- Prerequisites:
  - [CMake installation guide](https://cmake.org/install/).
  - [OpenCV installation guide](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html).
  - [MPI, preferably MPICH installation guide here](https://mpitutorial.com/tutorials/installing-mpich2/).
  - OpenCL
- After finishing up up with the prerequisites, try running the `setup.sh` script.
  - This will download and setup some graph datasets and other files required for testing.
- To build the projects, just run the `build.sh` script.
- To run/execute the projects, use the `run.sh` script. Just execute the `run.sh` 1st to get help on how to use the script.

---

## Reference materials

- OpenMP
  - [OpenMP tutorials](https://www.openmp.org/resources/tutorials-articles/)
  - [Tim Mattson's yt playlist](https://www.youtube.com/watch?v=nE-xN4Bf8XI&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG)
  - [Lawrence Livermore National Laboratory](https://hpc.llnl.gov/training/tutorials/openmp-tutorial)
  - [Jaka's corner](http://jakascorner.com/blog/)
- MPI
  - [Lawrence Livermore National Laboratory](https://hpc-tutorials.llnl.gov/mpi/)
  - [MPI Tutorial](https://mpitutorial.com/tutorials/)
- OpenCL
  - [Texas Instrumentation docs](http://downloads.ti.com/mctools/esd/docs/opencl/index.html)
  - [Leonardo Araujo Santos blog](https://leonardoaraujosantos.gitbook.io/opencl/)
  - [OpenCL 1.2 reference manual](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/)
- HPC
  - [UC Berkley CS 267](https://people.eecs.berkeley.edu/~demmel/cs267_Spr15/)
  - [UC Berkley CS 267 yt playlist](https://www.youtube.com/playlist?list=PLkFD6_40KJIyX8nEjk6oTLWohdVhjjP3X)
- FAQs
  - [Determine max global work group size based on device memory in OpenCL?](https://stackoverflow.com/questions/23017005/determine-max-global-work-group-size-based-on-device-memory-in-opencl)
  - [Questions about global and local work size](https://stackoverflow.com/questions/3957125/questions-about-global-and-local-work-size)
