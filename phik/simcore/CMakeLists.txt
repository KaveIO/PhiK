CMAKE_MINIMUM_REQUIRED(VERSION 3.14.4)

SET(MODNAME "phik_simulation_core")
SET(SUBPATH ${PROJECT_SOURCE_DIR}/phik/simcore/)

# ##############################################################################
#                          build ASA159 library                                #
# ##############################################################################
IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()

ADD_LIBRARY(_asa159
    OBJECT
    ${SUBPATH}/asa159.cpp
)

TARGET_INCLUDE_DIRECTORIES(_asa159 PRIVATE ${SUBPATH})
# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(${MODNAME}
    MODULE
        ${SUBPATH}/bindings.cpp
        ${SUBPATH}/simulation.hpp
        $<TARGET_OBJECTS:_asa159>
)

TARGET_COMPILE_DEFINITIONS(${MODNAME} PRIVATE VERSION_INFO=${PHIK_VERSION_INFO})

TARGET_INCLUDE_DIRECTORIES(${MODNAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${SUBPATH}>
)

INSTALL(TARGETS ${MODNAME} LIBRARY DESTINATION "phik/lib")
