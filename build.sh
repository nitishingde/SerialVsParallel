#!/usr/bin/env sh

if [ ! -d "build" ]; then
  mkdir build
fi
cd build

# DEBUG build
if [ $# -ge 1 ] && [ $1 = "--debug" ]; then
  if [ ! -d "debug" ]; then
    mkdir debug
  fi
  cd debug
  cmake ../../ -DCMAKE_BUILD_TYPE=Debug
  cmake --build ./ --target all
  cd ..
fi

# RELEASE build
if [ ! -d "release" ]; then
  mkdir release
fi
cd release
cmake ../../ -DCMAKE_BUILD_TYPE=Release
cmake --build ./ --target all
cd ..

cd ..
