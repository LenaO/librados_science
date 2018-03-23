add_library(radosdataset-shared SHARED rados_data.h rados_cache.h rados_data.cpp rados_cache.cpp)
add_library(radosdataset-static STATIC rados_data.h rados_cache.h  rados_data.cpp rados_cache.cpp)

SET_TARGET_PROPERTIES(radosdataset-shared PROPERTIES OUTPUT_NAME radosdataset CLEAN_DIRECT_OUTPUT 1)


add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/rados_data.h  ${CMAKE_SOURCE_DIR}/include)
add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/rados_cache.h  ${CMAKE_SOURCE_DIR}/include)

add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/libradosdataset.so  ${CMAKE_SOURCE_DIR}/lib)

add_custom_command(TARGET radosdataset-static POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/libradosdataset-static.a  ${CMAKE_SOURCE_DIR}/lib)



