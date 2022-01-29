CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

PROJECT(
    _phik_simulation
    VERSION ${PHIK_VERSION_INFO}
    DESCRIPTION "C++ bindings for simulation RXC tables"
    LANGUAGES CXX
)
SET(CMAKE_CXX_STANDARD 14)
# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

FIND_PACKAGE(pybind11 CONFIG REQUIRED)
INCLUDE(${PROJECT_SOURCE_DIR}/phik/simcore/CMakeLists.txt)

# Quiet a warning, since this project is only valid with SKBUILD
SET(ignoreMe "${SKBUILD}")
