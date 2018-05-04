project(lib_cudacomp_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(cudacomp SHARED cudacomp.c)


if(USE_MAGMA)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)
message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}") 
message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )   
target_include_directories(cudacomp PUBLIC ${MAGMA_INCLUDE_DIRS})
target_link_libraries(cudacomp PRIVATE ${MAGMA_LIBRARIES}) # DOES NOT SEEM TO BE INCLUDED IN COMPILE/LINK COMMAND
target_link_libraries(cudacomp PRIVATE cusolver)
set_target_properties(cudacomp PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_MAGMA")
target_compile_options(cudacomp PUBLIC -DUSE_MAGMA ${MAGMA_CFLAGS_OTHER})
endif(USE_MAGMA)


install(TARGETS cudacomp DESTINATION lib)
install(FILES cudacomp.h DESTINATION include)
