project(lib_cudacomp_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(cudacomp SHARED cudacomp.c)


if(USE_MAGMA)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)
message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}") # WORKS, LISTS DIRECTORIES
message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )   # WORKS, LISTS LIBRARIS
link_directories(${MAGMA_LIBRARY_DIRS} )                   # DOES NOT SEEM TO BE INCLUDED IN COMPILE/LINK COMMAND
target_link_libraries (cudacomp PUBLIC ${MAGMA_LIBRARIES}) # DOES NOT SEEM TO BE INCLUDED IN COMPILE/LINK COMMAND
set_target_properties(cudacomp PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_MAGMA")
endif(USE_MAGMA)


install(TARGETS cudacomp DESTINATION lib)
install(FILES cudacomp.h DESTINATION include)
