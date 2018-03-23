set( TESTS  writeDataSet.cpp  readDataSet.cpp CacheDictTest.cpp CacheTest.cpp )

set (LINK_LIBS ${LINK_LIBS} conduit "radosdataset-static" )

foreach( testsourcefile ${TESTS} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    # Make sure YourLib is linked to each app
    target_link_libraries( ${testname} ${LINK_LIBS} ${MY_RADOS} )
endforeach( testsourcefile ${APP_SOURCES} )



