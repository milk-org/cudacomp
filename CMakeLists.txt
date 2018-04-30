project(lib_cudacomp_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(cudacomp SHARED cudacomp.c)

install(TARGETS cudacomp DESTINATION lib)
install(FILES cudacomp.h DESTINATION include)

