CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

PROJECT(
    _phik_simulation
    VERSION 0.1.0
    DESCRIPTION "C++ bindings for simulation RXC tables"
    LANGUAGES CXX
)
SET(CMAKE_CXX_STANDARD 14)

IF (SKBUILD)
    # Scikit-Build does not add your site-packages to the search path automatically,
    # so we need to add it _or_ the pybind11 specific directory here.
    EXECUTE_PROCESS(
      COMMAND
          "${PYTHON_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _tmp_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
    )
    LIST(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
ENDIF ()

FIND_PACKAGE(pybind11 CONFIG REQUIRED)
INCLUDE(${PROJECT_SOURCE_DIR}/phik/simcore/CMakeLists.txt)
