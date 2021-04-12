#include "cuda.h"
#include "cuda_runtime.h"

#include "cusolver_symmetrize.h"

#define NB 64

__global__ void kern_fill_ssym_matrix(char src_uplo, float *data, int N,
                                      int ldda) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tidB;
  while (tid < N) {
    int row = tid / ldda;
    int col = tid - row * ldda;

    if (col > row) {
      tidB = row + col * ldda;
      if (src_uplo == 'U')
        data[tid] = data[tidB];
      else if (src_uplo == 'L')
        data[tidB] = data[tid];
    }
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void kern_fill_dsym_matrix(char src_uplo, double *data, int N,
                                      int ldda) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tidB;
  while (tid < N) {
    int row = tid / ldda;
    int col = tid - row * ldda;

    if (col > row) {
      tidB = row + col * ldda;
      if (src_uplo == 'U')
        data[tid] = data[tidB];
      else if (src_uplo == 'L')
        data[tidB] = data[tid];
    }
    tid += blockDim.x * gridDim.x;
  }
}

extern "C"
void ssymmetrize_lower(float *dA, int ldda) {
  if (ldda == 0) return;

  dim3 threads(NB);
  dim3 grid((ldda + NB - 1) / NB);
  int N = ldda * ldda;
  kern_fill_ssym_matrix<<<grid, threads, 0>>>('L', dA, N, ldda);
}

extern "C"
void dsymmetrize_lower(double *dA, int ldda) {
  if (ldda == 0) return;

  dim3 threads(NB);
  dim3 grid((ldda + NB - 1) / NB);
  int N = ldda * ldda;

  kern_fill_dsym_matrix<<<grid, threads, 0>>>('L', dA, N, ldda);
}