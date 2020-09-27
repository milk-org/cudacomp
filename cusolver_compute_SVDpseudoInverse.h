/** @file cusolver_compute_SVDpseudoInverse.h
 */
#ifndef CUSOLVER_COMPUTE_SVDPSEUDOINVERSE_H
#define CUSOLVER_COMPUTE_SVDPSEUDOINVERSE_H

#ifdef HAVE_CUDA

#include <cusolverDn.h>

#include "CommandLineInterface/CLIcore.h"

struct cusolver_compute_SVDpseudoInverse_data {
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;
  uint32_t loop_iter;
  uint8_t verbose;
  uint8_t comp_float;

  float *h_f_A;
  float *d_f_A;

  float *h_f_Ainv;
  float *d_f_Ainv;

  float *h_f_AtA;
  float *d_f_AtA;

  float *h_f_VT1;
  float *d_f_VT1;
  float *h_f_R;
  float *h_f_w;
  float *d_f_w;
  float *h_f_M2;
  float *d_f_M2;

  double *h_d_A;
  double *d_d_A;

  double *h_d_Ainv;
  double *d_d_Ainv;

  double *h_d_AtA;
  double *d_d_AtA;

  double *h_d_VT1;
  double *d_d_VT1;
  double *h_d_R;
  double *h_d_w;
  double *d_d_w;
  double *h_d_M2;
  double *d_d_M2;

  int *devInfo;
  float *d_f_work;
  double *d_d_work;
  int lwork;

  int info_gpu;
};

errno_t cusolver_compute_SVDpseudoInverse_addCLIcmd();

int CUDACOMP_cusolver_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name, const char *ID_Cmatrix_name, double SVDeps,
    long MaxNBmodes, const char *ID_VTmatrix_name, int LOOPmode, int PSINV_MODE,
    __attribute__((unused)) double qdwh_s,
    __attribute__((unused)) float qdwh_tol, int testmode);

#endif  // HAVE_CUDA

#endif  // CUSOLVER_COMPUTE_SVDPSEUDOINVERSE_H