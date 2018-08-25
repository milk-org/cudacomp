lib_LTLIBRARIES = libcudacomp.la
libcudacomp_la_SOURCES = cudacomp.c cudacomp.h
libcudacomp_la_SOURCES += cudacomp_MVMextractModesLoop.c

#AM_CFLAGS = $(CFLAGS)
#AM_CFLAGS += -fPIC
#AM_CFLAGS += $(MAGMA_CFLAGS)

#libcudacomp_la_LIBADD = $(MAGMA_LIBS)


AM_CPPFLAGS = -I@abs_top_srcdir@/src -lcuda -lcublas -lcudart -lcusolver
