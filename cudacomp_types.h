#ifndef _CUDACOMP_TYPES_H
#define _CUDACOMP_TYPES_H

#include "CommandLineInterface/CLIcore.h"

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/******************* CPU memory */
inline int testing_malloc_cpu_internal(void **ptr, size_t size, const char *var,
                                       const char *func, const char *file,
                                       int line) {
  if (posix_memalign(ptr, 64, size) != 0) {
    fprintf(stderr, "[%s:%d] (%s) testing_malloc_cpu failed for: %s\n", file,
            line, func, var);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define TESTING_MALLOC_CPU(ptr, type, size)                               \
  do {                                                                    \
    testing_malloc_cpu_internal((void **)&ptr, size * sizeof(type), #ptr, \
                                __FUNCTION__, __FILE__, __LINE__);        \
  } while (0)

#define TESTING_FREE_CPU(ptr) free(ptr)

/******************* Pinned CPU memory */
// In CUDA, this allocates pinned memory.
inline int testing_malloc_pin_internal(void **ptr, size_t size, const char *var,
                                       const char *func, const char *file,
                                       int line) {
  if (cudaSuccess != cudaHostAlloc(ptr, size, cudaHostAllocPortable)) {
    fprintf(stderr, "[%s:%d] (%s) testing_malloc_pin failed for: %s\n", file,
            line, func, var);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define TESTING_MALLOC_PIN(ptr, type, size)                               \
  do {                                                                    \
    testing_malloc_pin_internal((void **)&ptr, size * sizeof(type), #ptr, \
                                __FUNCTION__, __FILE__, __LINE__);        \
  } while (0)

inline int testing_free_pin_internal(void *ptr, const char *var,
                                     const char *func, const char *file,
                                     int line) {
  if (cudaSuccess != cudaFreeHost(ptr)) {
    fprintf(stderr, "[%s:%d] (%s) testing_free_pin failed for: %s\n", file,
            line, func, var);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define TESTING_FREE_PIN(ptr)                                               \
  do {                                                                      \
    testing_free_pin_internal(ptr, #ptr, __FUNCTION__, __FILE__, __LINE__); \
  } while (0)

/******************* GPU memory */
inline int testing_malloc_dev_internal(void **ptr, size_t size, const char *var,
                                       const char *func, const char *file,
                                       int line) {
  if (cudaSuccess != cudaMalloc(ptr, size)) {
    fprintf(stderr, "[%s:%d] (%s) testing_malloc_dev failed for: %s\n", file,
            line, func, var);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define TESTING_MALLOC_DEV(ptr, type, size)                               \
  do {                                                                    \
    testing_malloc_dev_internal((void **)&ptr, size * sizeof(type), #ptr, \
                                __FUNCTION__, __FILE__, __LINE__);        \
  } while (0)

inline int testing_free_dev_internal(void *ptr, const char *var,
                                     const char *func, const char *file,
                                     int line) {
  if (cudaSuccess != cudaFree(ptr)) {
    fprintf(stderr, "[%s:%d] (%s) testing_free_dev failed for: %s\n", file,
            line, func, var);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define TESTING_FREE_DEV(ptr)                                               \
  do {                                                                      \
    testing_free_dev_internal(ptr, #ptr, __FUNCTION__, __FILE__, __LINE__); \
  } while (0)

#endif  // HAVE_CUDA

// data passed to each thread
typedef struct {
  int thread_no;
  long numl0;
  int cindex;   // computation index
  int *status;  // where to white status

  // timers
  struct timespec *t0;
  struct timespec *t1;
  struct timespec *t2;
  struct timespec *t3;
  struct timespec *t4;
  struct timespec *t5;

} CUDACOMP_THDATA;

#ifdef HAVE_CUDA
/** \brief This structure holds the GPU computation setup for matrix
 * multiplication
 *
 * By declaring an array of these structures,
 * several parallel computations can be executed
 *
 */

typedef struct {
  int init;        /**< 1 if initialized               */
  int *refWFSinit; /**< reference init                 */
  int alloc;       /**< 1 if memory has been allocated */
  imageID CM_ID;
  uint64_t CM_cnt;
  long timerID;

  uint32_t M;
  uint32_t N;

  /// synchronization
  int sem; /**< if sem = 1, wait for semaphore to perform computation */
  int gpuinit;

  /// one semaphore per thread
  sem_t **semptr1; /**< command to start matrix multiplication (input) */
  sem_t **semptr2; /**< memory transfer to device completed (output)   */
  sem_t **semptr3; /**< computation done (output)                      */
  sem_t **semptr4; /**< command to start transfer to host (input)      */
  sem_t **semptr5; /**< output transfer to host completed (output)     */

  // computer memory (host)
  float *cMat;
  float **cMat_part;
  float *wfsVec;
  float **wfsVec_part;
  float *wfsRef;
  float **wfsRef_part;
  float *dmVec;
  float *dmVecTMP;
  float **dmVec_part;
  float **dmRef_part;

  // GPU memory (device)
  float **d_cMat;
  float **d_wfsVec;
  float **d_dmVec;
  float **d_wfsRef;
  float **d_dmRef;

  // threads
  CUDACOMP_THDATA *thdata;
  int *iret;
  pthread_t *threadarray;
  int NBstreams;
  cudaStream_t *stream;
  cublasHandle_t *handle;

  // splitting limits
  uint32_t *Nsize;
  uint32_t *Noffset;

  int *GPUdevice;

  int orientation;

  imageID IDout;

} GPUMATMULTCONF;
#endif  // HAVE_CUDA

#endif  // _CUDACOMP_TYPES_H
