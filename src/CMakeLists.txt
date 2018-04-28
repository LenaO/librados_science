add_library(radosdataset-shared SHARED rados_data.h rados_cache.h rados_hierarchy.h rados_data.cpp
rados_cache.cpp  rados_hierarchy.cpp  )
add_library(radosdataset-static STATIC rados_data.h rados_cache.h rados_hierarchy.h  rados_data.cpp
rados_cache.cpp  rados_hierarchy.cpp )

include_directories(${CONDUIT_INCLUDE_DIRS})
SET_TARGET_PROPERTIES(radosdataset-shared PROPERTIES OUTPUT_NAME radosdataset CLEAN_DIRECT_OUTPUT 1)


add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/rados_data.h  ${CMAKE_SOURCE_DIR}/include)
add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/rados_cache.h  ${CMAKE_SOURCE_DIR}/include)
add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/rados_hierarchy.h  ${CMAKE_SOURCE_DIR}/include)

add_custom_command(TARGET radosdataset-shared POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/libradosdataset.so  ${CMAKE_SOURCE_DIR}/lib)

add_custom_command(TARGET radosdataset-static POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/libradosdataset-static.a  ${CMAKE_SOURCE_DIR}/lib)



