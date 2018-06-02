project(lib_cudacomp_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(cudacomp SHARED cudacomp.c)


message("---- module cudacomp ------------------")

if(USE_MAGMA)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)
message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}") 
message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )   
message("---- MAGMA_CFLAGS_OTHER =  ${MAGMA_CFLAGS_OTHER}" ) 
target_include_directories(cudacomp PUBLIC ${MAGMA_INCLUDE_DIRS})
target_link_libraries(cudacomp PRIVATE ${MAGMA_LIBRARIES}) 
target_link_libraries(cudacomp PRIVATE cusolver)
set_target_properties(cudacomp PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_MAGMA")
target_compile_options(cudacomp PRIVATE -DUSE_MAGMA ${MAGMA_CFLAGS_OTHER})
endif(USE_MAGMA)

message("---------------------------------------")

install(TARGETS cudacomp DESTINATION lib)
install(FILES cudacomp.h DESTINATION include)



#target_compile_options(cudacomp PRIVATE -DUSE_MAGMA ${MAGMA_CFLAGS_OTHER} -Mcudalib=cublas)
#target_compile_options(cudacomp PUBLIC -DUSE_MAGMA -DNDEBUG -DADD_ ${OpenMP_C_FLAGS} -Mcudalib=cublas -ta=tesla:cc50,cuda9.1)

#-I/opt/pgi/linux86-64/2018/cuda/9.1/include/
#-L/opt/pgi/linux86-64/2018/cuda/9.1/lib64


#/opt/pgi/linux86-64/18.4/bin/pgcc -Mcudalib=cublas -I/data1/src/coffee/src -I/data1/src/coffee/src/CommandLineInterface/.. -I/data1/src/coffee/src/CommandLineInterface/src -I/data1/src/coffee/src/ImageStreamIO/src -I/data1/src/coffee/src/ImageStreamIO/.. -I/usr/local/magma/include -I/usr/local/cuda/include -I/opt/pgi/linux86-64/2018/cuda/9.1/include/ -L/opt/pgi/linux86-64/2018/cuda/9.1/lib64 -DPACKAGE_NAME=\"coffee\" -DPACKAGE_VERSION=\"0.0.0\" -DCONFIGDIR=\"/data1/src/coffee/config\" -DSOURCEDIR=\"/data1/src/coffee\" -DABSSRCTOPDIR=\"/data1/src/coffee\" -DPACKAGE_BUGREPORT=\"https://github.com/coffee-org/coffee/issues\" -DHAVE_CUDA -DHAVE_MAGMA -o CMakeFiles/coffee.dir/src/CLImain.c.o   -c /data1/src/coffee/src/CLImain.c
