cmake_minimum_required (VERSION 2.8)
project (Rado_DataSet)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_COMPILER icpc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_library(MY_RADOS NAMES rados
            PATHS   "/usr/lib"
                    "/usr/lib64"
                    "/usr/local/lib")

message("Do I found Rados ?" ${MY_RADOS})


find_package(PythonInterp)
find_package(PythonLibs)
find_package(NumPy)


if (${NUMPY_FOUNS})
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${MPI_C_COMPILE_FLAGS} -DHAVE_NUMPY" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -DHAVE_NUMPY")
    set (LINK_LIBS ${LINK_LIBS}  ${NUMPY_LIBS})
    include_directories(${NUMPY_INCLUDES})
endif()

include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)

add_subdirectory(test)

