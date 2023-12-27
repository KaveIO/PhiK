cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSKBUILD_PROJECT_NAME="phik" \
    -DSKBUILD_PROJECT_VERSION="0.12.4" \
    -DPHIK_MBUILD=ON \
    -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)') \
    -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build --target install --config Release --parallel 4
