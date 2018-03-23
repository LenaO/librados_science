cmake_minimum_required (VERSION 2.8)
project (Rado_DataSet)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_COMPILER icpc)


find_library(MY_RADOS NAMES rados
            PATHS   "/usr/lib"
                    "/usr/lib64"
                    "/usr/local/lib")

message("Do I found Rados ?" ${MY_RADOS})


include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)

add_subdirectory(test)

