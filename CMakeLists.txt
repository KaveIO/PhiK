cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "C++ bindings for simulation RXC tables"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

find_package(pybind11 CONFIG REQUIRED)
set(SUBPATH ${PROJECT_SOURCE_DIR}/phik/simcore/)

# ##############################################################################
# build ASA159 library                                #
# ##############################################################################
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_library(_asa159 OBJECT ${SUBPATH}/asa159.cpp)

target_include_directories(_asa159 PRIVATE ${SUBPATH})
# ##############################################################################
# EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(_phik_simulation_core MODULE ${SUBPATH}/bindings.cpp
                    ${SUBPATH}/simulation.hpp $<TARGET_OBJECTS:_asa159>)

target_compile_definitions(_phik_simulation_core
                           PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})

target_include_directories(
  _phik_simulation_core PUBLIC $<INSTALL_INTERFACE:include>
                               $<BUILD_INTERFACE:${SUBPATH}>)
if(PHIK_MBUILD)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
endif()

install(TARGETS _phik_simulation_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")

# Quiet a warning, since this project is only valid with SKBUILD
set(ignoreMe "${SKBUILD}")
