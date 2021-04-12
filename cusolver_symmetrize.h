/** @file cusolver_symmetrize.h
 */
#ifndef CUSOLVER_SYMMETRIZE
#define CUSOLVER_SYMMETRIZE

#ifdef HAVE_CUDA

#include "cuda.h"
#include "cuda_runtime.h"

extern "C"
void ssymmetrize_lower(float *dA, int ldda);

extern "C"
void dsymmetrize_lower(double *dA, int ldda);

#endif  // HAVE_CUDA

#endif  // CUSOLVER_SYMMETRIZE