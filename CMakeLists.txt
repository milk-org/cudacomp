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
target_compile_options(cudacomp PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA ${MAGMA_CFLAGS_OTHER})
endif(USE_MAGMA)

if(USE_QDWHPARTIAL)
message("---- QDWHPARTIAL_INCLUDE_PATH =  ${QDWHPARTIAL_INCLUDE_PATH}") 
message("---- QDWHPARTIAL_LIBRARIES    =  ${QDWHPARTIAL_LIBRARIES}" )   
target_include_directories(cudacomp PUBLIC ${QDWHPARTIAL_INCLUDE_PATH})
target_link_libraries(cudacomp PRIVATE ${QDWHPARTIAL_LIBRARIES})
set_target_properties(cudacomp PROPERTIES COMPILE_FLAGS "-DHAVE_QDWHpartial")
target_compile_options(cudacomp PUBLIC -DUSE_QDWHPARTIAL)
endif(USE_QDWHPARTIAL)

message("---------------------------------------")

install(TARGETS cudacomp DESTINATION lib)
install(FILES cudacomp.h DESTINATION include)



