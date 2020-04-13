# library name
set(LIBNAME "milkcudacomp")


set(SOURCEFILES
	cudacomp.c
	cudacomp_MVMextractModesLoop.c)

set(INCLUDEFILES
	image_gen.h)





# DEFAULT SETTINGS 
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")


# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
# 
add_library(${LIBNAME} SHARED ${SOURCEFILES})



message("---- module cudacomp ------------------")

if(USE_CUDA)
find_package(CUDA 9.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
message("---- CUDA_INCLUDE_DIRS   = ${CUDA_INCLUDE_DIRS}")
message("---- CUDA_LIBRARIES      = ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY}" )   
message("---- CUDA_CUBLAS_LIBRARIES    = ${CUDA_CUBLAS_LIBRARIES}" )   
target_include_directories(${LIBNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY}) 
target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)



if(USE_MAGMA)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)
message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}") 
message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )   
message("---- MAGMA_CFLAGS_OTHER =  ${MAGMA_CFLAGS_OTHER}" ) 
target_include_directories(${LIBNAME} PUBLIC ${MAGMA_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE ${MAGMA_LIBRARIES})
target_compile_options(${LIBNAME} PRIVATE ${MAGMA_CFLAGS_OTHER})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
#target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
endif(USE_MAGMA)



if(USE_QDWHPARTIAL)
message("---- QDWHPARTIAL_INCLUDE_PATH =  ${QDWHPARTIAL_INCLUDE_PATH}") 
message("---- QDWHPARTIAL_LIBRARIES    =  ${QDWHPARTIAL_LIBRARIES}" )   
target_include_directories(${LIBNAME} PUBLIC ${QDWHPARTIAL_INCLUDE_PATH})
target_link_libraries(${LIBNAME} PRIVATE ${QDWHPARTIAL_LIBRARIES})
set_target_properties(${LIBNAME} PROPERTIES COMPILE_FLAGS "-DHAVE_QDWHpartial")
target_compile_options(${LIBNAME} PUBLIC -DUSE_QDWHPARTIAL)
endif(USE_QDWHPARTIAL)

message("---------------------------------------")


install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include)










