lib_LTLIBRARIES = libcudacomp.la
libcudacomp_la_SOURCES = cudacomp.c cudacomp.h

#AM_CFLAGS = $(CFLAGS)
#libcudacomp_la_LIBADD = -lcuda -lcublas -lcudart -lcusolver
AM_CPPFLAGS = -I@abs_top_srcdir@/src -lcuda -lcublas -lcudart -lcusolver
#AM_CPPFLAGS = -I@abs_top_srcdir@/src -fopenmp -lcuda -lcublas -lcudart -lcusolver
