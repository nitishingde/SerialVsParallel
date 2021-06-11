if [ -d "build/debug" ]; then
  cmake --build ./build/debug --target clean
  echo "Cleaned build/debug"
fi

if [ -d "build/release" ]; then
  cmake --build ./build/release --target clean
  echo "Cleaned build/release"
fi
