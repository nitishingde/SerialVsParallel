#!/usr/bin/env sh

if [ ! -d "build/release" ]; then
  echo "Need to run the \"${PWD}/build.sh\" script first."
  exit 0
fi

usage='sh ./run.sh <id>
1.1> Pi without mpi
1.2> Pi with mpi
2.1> Prime without mpi
2.2> Prime with mpi
  3> Matrix Multiplication
  4> Image Processing
  5> Graph
'

if [ $# -eq 0 ]; then
  printf '%s' "$usage"
  exit 0
fi

commandArgFound=false

for option in $@ ; do
  case $option in
  1.1)
    echo "> Pi without mpi"
    build/release/Pi
    commandArgFound=true
    ;;
  1.2|1)
    echo "> Pi with mpi"
    mpiexec.mpich -np 2 build/release/Pi --use-mpi
    commandArgFound=true
    ;;
  2.1)
    echo "> Prime without mpi"
    build/release/Prime
    commandArgFound=true
    ;;
  2.2|2)
    echo "> Prime with mpi"
    mpiexec.mpich -np 2 build/release/Prime --use-mpi
    commandArgFound=true
    ;;
  3)
    echo "> Matrix Multiplication"
    build/release/MatrixMultiplication
    commandArgFound=true
    ;;
  4)
    echo "> Image Processing"
    build/release/ImageProcessing
    commandArgFound=true
    ;;
  5)
    echo "> Graph"
    build/release/Graph
    commandArgFound=true
    ;;
  esac
done

if ! $commandArgFound; then
  printf '%s' "$usage"
fi