/** @file magma_compute_SVDpseudoInverse.c
 */

#ifdef HAVE_CUDA

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include "cusolver_compute_SVDpseudoInverse.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "CommandLineInterface/timeutils.h"
#include "cudacomp_types.h"
#include "cusolver_symmetrize.h"

static struct cusolver_compute_SVDpseudoInverse_data CUSOLVER_DATA = {
    .loop_iter = 0,
    .verbose = 1,
    .comp_float = 1,

    .h_f_A = NULL,
    .d_f_A = NULL,

    .h_f_Ainv = NULL,
    .d_f_Ainv = NULL,

    .h_f_AtA = NULL,
    .d_f_AtA = NULL,

    .h_f_VT1 = NULL,
    .d_f_VT1 = NULL,
    .h_f_R = NULL,
    .h_f_w = NULL,
    .d_f_w = NULL,
    .d_f_M2 = NULL,

    .h_d_A = NULL,
    .d_d_A = NULL,

    .h_d_Ainv = NULL,
    .d_d_Ainv = NULL,

    .h_d_AtA = NULL,
    .d_d_AtA = NULL,

    .h_d_VT1 = NULL,
    .d_d_VT1 = NULL,
    .h_d_R = NULL,
    .h_d_w = NULL,
    .d_d_w = NULL,
    .d_d_M2 = NULL,

    .devInfo = NULL,
    .d_f_work = NULL,
    .d_d_work = NULL,
    .lwork = 0,

    .info_gpu = 0};

#define CHECK_CUSOLVER(fct)                                             \
  do {                                                                  \
    cusolverStatus_t info = fct;                                        \
    if (info != CUSOLVER_STATUS_SUCCESS) {                              \
      printf("%s@%d cuSolver returned error %d.\n", __FILE__, __LINE__, \
             (int)info);                                                \
    }                                                                   \
  } while (0)

// ==========================================
// Forward declaration(s)
// ==========================================

int CUDACOMP_cusolver_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name, const char *ID_Cmatrix_name, double SVDeps,
    long MaxNBmodes, const char *ID_VTmatrix_name, int LOOPmode, int PSINV_MODE,
    __attribute__((unused)) double qdwh_s,
    __attribute__((unused)) float qdwh_tol, int testmode);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t CUDACOMP_cusolver_compute_SVDpseudoInverse_cli() {
  if (CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
          CLI_checkarg(4, 2) + CLI_checkarg(5, 3) + CLI_checkarg(6, 2) +
          CLI_checkarg(7, 1) + CLI_checkarg(8, 1) ==
      0) {
    CUDACOMP_cusolver_compute_SVDpseudoInverse(
        data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.numf, data.cmdargtoken[4].val.numl,
        data.cmdargtoken[5].val.string, 0, data.cmdargtoken[6].val.numl,
        data.cmdargtoken[7].val.numf, data.cmdargtoken[8].val.numf, 0);

    return CLICMD_SUCCESS;
  } else {
    return CLICMD_INVALID_ARG;
  }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t cusolver_compute_SVDpseudoInverse_addCLIcmd() {
#ifndef USE_MAGMA
  RegisterCLIcommand(
      "cudacomppsinv", __FILE__, CUDACOMP_cusolver_compute_SVDpseudoInverse_cli,
      "compute pseudo inverse",
      "<input matrix [string]> <output pseudoinv [string]> <eps [float]> "
      "<NBmodes [long]> <VTmat [string]>",
      "cudacomppsinv matA matAinv 0.01 100 VTmat 0 1e-4 1e-7",
      "int CUDACOMP_cusolver_compute_SVDpseudoInverse(const char "
      "*ID_Rmatrix_name, const char *ID_Cmatrix_name, double SVDeps, long "
      "MaxNBmodes, const char *ID_VTmatrix_name, int LOOPmode, int PSINV_MODE, "
      "double qdwh_s, float qdwh_tol)");
#endif
  RegisterCLIcommand(
      "cudacomppsinv_cusolver", __FILE__,
      CUDACOMP_cusolver_compute_SVDpseudoInverse_cli, "compute pseudo inverse",
      "<input matrix [string]> <output pseudoinv [string]> <eps [float]> "
      "<NBmodes [long]> <VTmat [string]>",
      "cudacomppsinv matA matAinv 0.01 100 VTmat 0 1e-4 1e-7",
      "int CUDACOMP_cusolver_compute_SVDpseudoInverse(const char "
      "*ID_Rmatrix_name, const char *ID_Cmatrix_name, double SVDeps, long "
      "MaxNBmodes, const char *ID_VTmatrix_name, int LOOPmode, int PSINV_MODE, "
      "double qdwh_s, float qdwh_tol)");

  return RETURN_SUCCESS;
}

/**
 *  @brief Computes matrix pseudo-inverse (AT A)^-1 AT, using
 *eigenvector/eigenvalue decomposition of AT A
 *
 *
 * Computes pseuso inverse of a matrix.\n
 * Column-major representation used to match magma and lapack.\n
 * When viewed as an image, matrix leading dimension is size[0] = horizontal
 *axis. When viewed in an image viewer, the first column is on the bottom side
 *with the first element in bottom left corner, so the matrix appears rotated
 *counter-clockwise by 90deg from its conventional representation where first
 *column is on the left side.\n Returns transpose of pseudoinverse.\n
 *
 *
 *
 * ## Matrix representation details
 *
 * Using column-major indexing\n
 * When viewed as a FITS file, the first matrix column (= vector) appears as the
 *bottom line of the FITS image.\n First matrix element is bottom left corner,
 *second element is immediately to the right of it.
 *
 * Noting elements as a[row,column] = a[i,j], elements are accessed as in memory
 *as: a[ j * M + i ]
 *
 * FITS file representation (ds9 view) starts from bottom left corner.
 *
 * 		a[000,N-1] -> a[001,N-1] -> ... -> a[M-1,N-1]
 * 		............................................. ^
 * 		a[000,001] -> a[001,001] -> ... -> a[M-1,001] ^
 * 		a[000,000] -> a[001,000] -> ... -> a[M-1,000] ^     : this is
 *the first matrix row
 *
 * Note that a tall input matrix (M>N) will appear short if viewed as an image.
 * To view the FITS file in the conventional matrix view, rotate by 90 deg
 *clockwise.
 *
 *
 *
 * ## Application Notes
 *
 *  Use LOOPmode = 1 for computing the same size SVD, same input and output
 *location
 *
 * ### Use case: Response matrix to compute control matrix
 *
 * When using function to invert AO response matrix with AOloopControl module,
 *input is 2D or 3D image: M: number of sensors    (AO control) =  size[0] (2D)
 *= size[0]*size[1] (3D) N: number of actuators  (AO control) =  size[1] (2D) =
 *size[2] (3D)
 *
 * 	We assume M>N
 *
 *
 * ### Use case: Predictive control
 *
 * When using function to compute pseudo-inverse of data matrix (predictive
 *control), input matrix is a 2D image which is the Transpose of the data
 *matrix. M: number of measurements samples  = size[0] (2D) N: dimension of each
 *measurement   = size[1] (2D)
 *
 * We assume M>N
 *
 *
 *
 *
 * ## Algorithm details and main computation steps
 *
 * Notations:
 * 	AT is transpose of A
 * 	A+ is pseudo inverse of A
 *
 *  Computes pseudo-inverse : A+ = (AT A)^-1 AT
 *  Inverse of AT A is computed by SVD
 *
 * SVD:   A = U S V^T
 *   U are eigenvectors of A A^T
 *   V are eigenvectors of A^T A, computed at step 4 below
 *
 * Linear algebra reminder: equivalence between (AT A)^-1 AT and V S^-1 UT
 *
 * Definition of pseudoinverse:
 * A+ = (AT A)^-1 AT
 * singular value decomposition of A = U S VT
 * A+ = ( V S UT U S VT )^-1 V S UT
 * Since U is unitary, UT U = Id ->
 * A+ = ( V S^2 VT )^-1 V S UT
 * A+ = VT^-1 S^-2 V^-1 V S UT
 * A+ = V S^-1 UT
 *
 *  Main steps (non-QDWH):
 *
 *  STEP 1 :   Fill input data into CUSOLVER_DATA.h_f_A on host
 *
 *  STEP 2 :   Copy input data to GPU                                 ->
 *CUSOLVER_DATA.d_f_A        (MxN matrix on device)
 *
 *  STEP 3 :   Compute  trans(A) x A   : CUSOLVER_DATA.d_f_A x
 *CUSOLVER_DATA.d_f_A      -> CUSOLVER_DATA.d_f_AtA      (NxN matrix
 *on device)
 *
 *  STEP 4 :   Compute eigenvalues and eigenvectors of A^T A          ->
 *CUSOLVER_DATA.d_f_AtA      (NxN matrix on device) Calls magma_ssyevd_gpu
 *: Compute the eigenvalues and optionally eigenvectors of a symmetric real
 *matrix in single precision, GPU interface, big matrix. This function computes
 *in single precision all eigenvalues and, optionally, eigenvectors of a real
 *symmetric matrix A defined on the device. The  first parameter can take the
 *values MagmaVec,'V' or MagmaNoVec,'N' and answers the question whether the
 *eigenvectors are desired. If the eigenvectors are desired, it uses a divide
 *and conquer algorithm.  The symmetric matrix A can be stored in lower
 *(MagmaLower,'L') or upper  (MagmaUpper,'U') mode. If the eigenvectors are
 *desired, then on exit A contains orthonormal eigenvectors. The eigenvalues are
 *stored in an array w
 *
 *  STEP 5 :   Set eigenvalue limit
 *
 *  STEP 6 :   Write eigenvectors to V^T matrix
 *
 *  STEP 7 :   Write eigenvectors/eigenvalue to CUSOLVER_DATA.h_d_VT1 if
 *eigenvalue > limit Copy to CUSOLVER_DATA.d_d_VT1
 *
 *  STEP 8 :   Compute M2 = VT1 VT. M2 is (AT A)^-1
 *
 *  STEP 9 :   Compute Ainv = M2 A. This is the pseudo inverse
 *
 * @note SVDeps^2 is applied as a limit to the eigenvectors of AT A, which are
 *equal to the squares of the singular values of A, so this is equivalent to
 *applying SVDeps as a limit on the singular values of A
 * @note When used to compute AO control matrix, N=number of actuators/modes,
 *M=number of WFS elements
 * @note EIGENVALUES are good to about 1e-6 of peak eigenvalue in single
 *precision, much better with double precision
 * @note 2.5x faster in single precision
 *
 * @note If provided with an additional data matrix named "", an additional
 *Matrix Matrix product between Ainv and the provided matrix will be performed.
 *This feature is used for predictive control and sensor fusion to create a
 *control matrix.
 *
 * TEST MODE OUTPOUT
 *
 * non-QDWH mode:
 *
 * test_mA.fits               content of CUSOLVER_DATA.h_f_A
 * test_mAtA.fits             content of transpose(A) x A =
 *CUSOLVER_DATA.d_f_AtA (output of STEP 3) test_eigenv.dat            list
 *of eigenvalues test_SVDmodes.log number of singular values kept test_mM2.fits
 *matrix M2 (output of STEP 8) test_mVT.fits              matrix transpose(V) =
 *eigenvectors (output of step 6) test_mAinv.fits            transpose of
 *pseudoinverse test_AinvA.fits            product of Ainv with A, should be
 *close to identity matrix size NxN
 *
 *
 * QDWH mode:
 *
 * test_mA.QDWH.fits          content of CUSOLVER_DATA.h_f_A
 * test_Aorig.QDWH.txt        content of CUSOLVER_DATA.h_f_A prior to
 *calling psinv function test_sv.QDWH.dat           singular values after call
 *to psinv function test_SVDmodes.QDWH.log     number of singular values kept
 *(note : independent form pseudo-inverse computation) test_mAinv.QDWH.fits
 *transpose of pseudoinverse test_AinvA.QDWH.fits       product of Ainv with A,
 *should be close to identity matrix size NxN
 */

int CUDACOMP_cusolver_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name, const char *ID_Cmatrix_name, double SVDeps,
    long MaxNBmodes, const char *ID_VTmatrix_name, int LOOPmode,
    __attribute__((unused)) int PSINV_MODE,
    __attribute__((unused)) double qdwh_s,
    __attribute__((unused)) float qdwh_tol, int testmode) {
  long ID_Rmatrix;
  uint8_t datatype;
  uint32_t *arraysizetmp;
  int N, M;
  long ii, jj, k;
  int info;

  imageID ID_PFfmdat = -1;  // optional final M-M product
  imageID ID_AtA, ID_VT, ID_Ainv;

  /// Timing tests
  // int timing = 1;                                                        /**<
  // 1 if timing test ON, 0 otherwise */
  struct timespec t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12,
      t13; /**< Timers                           */
  double t01d, t12d, t23d, t34d, t45d, t56d, t67d, t78d, t89d, t910d, t1011d,
      t1112d, t1213d, t013d; /**< Times in sec                     */

  FILE *fp;
  char fname[200];

  long MaxNBmodes1, mode;

  imageID ID_Cmatrix;

  // TESTING FLAGS
  CUSOLVER_DATA.verbose = 1;

  /**< 1 if single precision, 0 if double precision */
  CUSOLVER_DATA.comp_float = 1;

  int mout;

  int dAinvMODE = 0;

  //  if(timing==1)
  clock_gettime(CLOCK_REALTIME, &t0);

  /**
   *
   *
   * MATRIX REPRESENTATION CONVENTION
   *

   *
   */

  ///
  /// MAGMA uses column-major matrices. For matrix A with dimension (M,N),
  /// element A(i,j) is A[ j*M + i] i = 0 ... M : row index, coefficient of a
  /// vector j = 0 ... N : column index, vector index M is the matrix leading
  /// dimension = lda M = number of rows N = number of columns (assuming here
  /// that vector = column of the matrix)
  ///

  arraysizetmp = (uint32_t *)malloc(sizeof(uint32_t) * 3);

  ID_Rmatrix = image_ID(ID_Rmatrix_name);
  datatype = data.image[ID_Rmatrix].md[0].datatype;

  if (data.image[ID_Rmatrix].md[0].naxis == 3) {
    /// each column (N=cst) of A is a z=cst slice of image Rmatrix
    M = data.image[ID_Rmatrix].md[0].size[0] *
        data.image[ID_Rmatrix].md[0].size[1];

    N = data.image[ID_Rmatrix].md[0].size[2];

    if (CUSOLVER_DATA.verbose == 1) {
      printf("3D image -> %ld %ld\n", (long)M, (long)N);
      fflush(stdout);
    }
  } else {
    /// each column (N=cst) of A is a line (y=cst) of Rmatrix (90 deg rotation)
    M = data.image[ID_Rmatrix].md[0].size[0];

    N = data.image[ID_Rmatrix].md[0].size[1];

    if (CUSOLVER_DATA.verbose == 1) {
      printf("2D image -> %ld %ld\n", (long)M, (long)N);
      fflush(stdout);
    }
  }

  // TEST
  // for(ii=0;ii<N;ii++)
  // data.image[ID_Rmatrix].array.F[ii*M+ii] += 1.0;

  if (CUSOLVER_DATA.verbose == 1) {
    printf("cusolver :    M = %ld , N = %ld\n", (long)M, (long)N);
    fflush(stdout);
  }

  // =================================================================
  //             MEMORY ALLOCATION
  //
  // ----- QSWHpartial --------
  // CUSOLVER_DATA.h_A
  // CUSOLVER_DATA.d_A
  // CUSOLVER_DATA.h_S
  // CUSOLVER_DATA.d_U
  // CUSOLVER_DATA.d_VT
  // CUSOLVER_DATA.d_B
  //
  // ----- std magma ----------
  // CUSOLVER_DATA.h_A
  // CUSOLVER_DATA.d_A
  // CUSOLVER_DATA.h_AtA
  // CUSOLVER_DATA.d_AtA
  // CUSOLVER_DATA.h_VT1
  // CUSOLVER_DATA.d_VT1
  // CUSOLVER_DATA.d_M2
  //
  // =================================================================

  if (CUSOLVER_DATA.verbose == 1) {
    printf("ALLOCATION FOR PSINV CUSOLVER M=%ld N=%ld\n", (long)M, (long)N);
    fflush(stdout);
  }

  if (CUSOLVER_DATA.loop_iter == 0)  /// memory is only allocated on first pass
  {
    if (CUSOLVER_DATA.verbose == 1) {
      printf("INITIALIZE CUBLAS & CUSOLVER\n");
      fflush(stdout);
    }
    cublasCreate(&CUSOLVER_DATA.cublas_handle);
    CHECK_CUSOLVER(cusolverDnCreate(&CUSOLVER_DATA.cusolver_handle));

    TESTING_MALLOC_DEV(CUSOLVER_DATA.devInfo, int, 1);

    if (CUSOLVER_DATA.comp_float == 1) {
      // get workspace size
      cusolverDnSsyevd_bufferSize(
          CUSOLVER_DATA.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, N, CUSOLVER_DATA.d_f_A, N,
          CUSOLVER_DATA.d_f_w, &CUSOLVER_DATA.lwork);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_A, float, M *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_A, float, M *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_AtA, float, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_AtA, float, N *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_VT1, float, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_VT1, float, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_M2, float, N *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_Ainv, float, N *M);

      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_work, float, CUSOLVER_DATA.lwork);
      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_w, float, N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_w, float, N);
      TESTING_MALLOC_PIN(CUSOLVER_DATA.h_f_R, float, N *N);
    } else {
      // get workspace size
      cusolverDnDsyevd_bufferSize(
          CUSOLVER_DATA.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, N, CUSOLVER_DATA.d_d_A, N,
          CUSOLVER_DATA.d_d_w, &CUSOLVER_DATA.lwork);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_A, double, M *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_A, double, M *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_AtA, double, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_AtA, double, N *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_VT1, double, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_VT1, double, N *N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_M2, double, N *N);

      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_Ainv, double, N *M);

      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_work, double, CUSOLVER_DATA.lwork);
      TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_w, double, N);
      TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_w, double, N);
      TESTING_MALLOC_PIN(CUSOLVER_DATA.h_d_R, double, N *N);
    }
  }

  if (CUSOLVER_DATA.verbose == 1) {
    printf("CUSOLVER READY\n");
    fflush(stdout);
  }

  clock_gettime(CLOCK_REALTIME, &t1);

  // ****************************************************
  // STEP 1 :   Fill input data into CUSOLVER_DATA.h_f_A on host
  // ****************************************************
  // magma array is column-major.
  //

  if (datatype == _DATATYPE_FLOAT) {
    if (CUSOLVER_DATA.comp_float == 1) {
      // need CUSOLVER_DATA.h_f_A, otherwise, straight to
      // CUSOLVER_DATA.d_f_A
      if ((testmode == 1)) {
        memcpy(CUSOLVER_DATA.h_f_A, data.image[ID_Rmatrix].array.F,
               sizeof(float) * M * N);
        // copy from host to device
        cublasSetMatrix(M, N, sizeof(float), CUSOLVER_DATA.h_f_A, M,
                        CUSOLVER_DATA.d_f_A, M);

      } else {
        cublasSetMatrix(M, N, sizeof(float), data.image[ID_Rmatrix].array.F, M,
                        CUSOLVER_DATA.d_f_A, M);
      }
    } else {
      for (ii = 0; ii < M * N; ii++) {
        CUSOLVER_DATA.h_d_A[ii] = data.image[ID_Rmatrix].array.F[ii];
      }

      // copy from host to device
      cublasSetMatrix(M, N, sizeof(double), CUSOLVER_DATA.h_d_A, M,
                      CUSOLVER_DATA.d_d_A, M);
    }
  } else {
    if (CUSOLVER_DATA.comp_float == 1) {
      for (ii = 0; ii < M * N; ii++) {
        CUSOLVER_DATA.h_f_A[ii] = data.image[ID_Rmatrix].array.D[ii];
      }

      // copy from host to device
      cublasSetMatrix(M, N, sizeof(float), CUSOLVER_DATA.h_f_A, M,
                      CUSOLVER_DATA.d_f_A, M);
    } else {
      if (testmode == 1)  // need CUSOLVER_DATA.h_d_A for testing
      {
        // for(ii=0; ii<M*N; ii++)
        //    CUSOLVER_DATA.h_d_A[ii] = data.image[ID_Rmatrix].array.D[ii];
        memcpy(CUSOLVER_DATA.h_d_A, data.image[ID_Rmatrix].array.D,
               sizeof(double) * M * N);
        // copy from host to device
        cublasSetMatrix(M, N, sizeof(double), CUSOLVER_DATA.h_d_A, M,
                        CUSOLVER_DATA.d_d_A, M);
      } else {
        cublasSetMatrix(M, N, sizeof(double), data.image[ID_Rmatrix].array.D, M,
                        CUSOLVER_DATA.d_d_A, M);
      }
    }
  }

  if (testmode == 1) {
    long ID_A;

    ID_A = create_2Dimage_ID("mA", M, N);
    if (CUSOLVER_DATA.comp_float == 1) {
      for (ii = 0; ii < M * N; ii++) {
        data.image[ID_A].array.F[ii] = CUSOLVER_DATA.h_f_A[ii];
      }
    } else {
      for (ii = 0; ii < M * N; ii++) {
        data.image[ID_A].array.F[ii] = CUSOLVER_DATA.h_d_A[ii];
      }
    }

    save_fits("mA", "!test_mA.QDWH.fits");

    delete_image_ID("mA");
  }

  // ****************************************************
  // STEP 2 :   Copy input data from CPU to GPU
  // ****************************************************

  if (CUSOLVER_DATA.verbose == 1) {
    printf("CUSOLVER: OFFLOAD TO THE GPU\n");
    fflush(stdout);
  }

  // copy from host to device
  //
  {
    // START STD MAGMA ===============================================

    clock_gettime(CLOCK_REALTIME, &t2);

    if (CUSOLVER_DATA.verbose == 1) {
      printf("CUBLAS: COMPUTE trans(A) x A\n");
      fflush(stdout);
    }

    // ****************************************************
    // STEP 3 :   Compute trans(A) x A    : CUSOLVER_DATA.d_f_A x
    // CUSOLVER_DATA.d_f_A      -> CUSOLVER_DATA.d_f_AtA      (NxN
    // matrix on device)
    // ****************************************************

    if (CUSOLVER_DATA.comp_float == 1) {
      float alpha = 1, beta = 0;
      cublasSsyrk(CUSOLVER_DATA.cublas_handle, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_T, N, M, &alpha, CUSOLVER_DATA.d_f_A, M, &beta,
                  CUSOLVER_DATA.d_f_AtA, N);
      ssymmetrize_lower(CUSOLVER_DATA.d_f_AtA, N);
    } else {
      double alpha = 1, beta = 0;
      cublasDsyrk(CUSOLVER_DATA.cublas_handle, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_T, N, M, &alpha, CUSOLVER_DATA.d_d_A, M, &beta,
                  CUSOLVER_DATA.d_d_AtA, N);
      dsymmetrize_lower(CUSOLVER_DATA.d_d_AtA, N);
    }

    if (testmode == 1) {
      // copy from GPU to CPU
      if (CUSOLVER_DATA.comp_float == 1) {
        cublasGetMatrix(N, N, sizeof(float), CUSOLVER_DATA.d_f_AtA, N,
                        CUSOLVER_DATA.h_f_AtA, N);
      } else {
        cublasGetMatrix(N, N, sizeof(double), CUSOLVER_DATA.d_d_AtA, N,
                        CUSOLVER_DATA.h_d_AtA, N);
      }

      ID_AtA = create_2Dimage_ID("mAtA", N, N);
      if (CUSOLVER_DATA.comp_float == 1) {
        for (ii = 0; ii < N * N; ii++) {
          data.image[ID_AtA].array.F[ii] = CUSOLVER_DATA.h_f_AtA[ii];
        }
      } else {
        for (ii = 0; ii < N * N; ii++) {
          data.image[ID_AtA].array.F[ii] = CUSOLVER_DATA.h_d_AtA[ii];
        }
      }
      save_fits("mAtA", "!test_mAtA.fits");
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t3);

    // ****************************************************
    // STEP 4 :   Compute eigenvalues and eigenvectors of AT A   ->
    // CUSOLVER_DATA.d_f_AtA (NxN matrix on device)
    //
    // SVD of AT A = V S^2 VT
    // calls function magma_ssyevd_gpu
    //
    //
    // ****************************************************

    if (CUSOLVER_DATA.verbose == 1) {
      printf("COMPUTE eigenvalues and eigenvectors of AT A\n");
      fflush(stdout);
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t4);

    if (CUSOLVER_DATA.comp_float == 1) {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> FLOAT cusolverDnSsyevd -> ");
        fflush(stdout);
      }

      CHECK_CUSOLVER(cusolverDnSsyevd(
          CUSOLVER_DATA.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, N, CUSOLVER_DATA.d_f_AtA, N,
          CUSOLVER_DATA.d_f_w, CUSOLVER_DATA.d_f_work, CUSOLVER_DATA.lwork,
          CUSOLVER_DATA.devInfo));

      cudaMemcpy(CUSOLVER_DATA.h_f_w, CUSOLVER_DATA.d_f_w, N * sizeof(float),
                 cudaMemcpyDeviceToHost);

      if (CUSOLVER_DATA.verbose == 1) {
        printf(" DONE\n");
        fflush(stdout);
      }
    } else {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> DOUBLE cusolverDnDsyevd -> ");
        fflush(stdout);
      }
      CHECK_CUSOLVER(cusolverDnDsyevd(
          CUSOLVER_DATA.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, N, CUSOLVER_DATA.d_d_AtA, N,
          CUSOLVER_DATA.d_d_w, CUSOLVER_DATA.d_d_work, CUSOLVER_DATA.lwork,
          CUSOLVER_DATA.devInfo));

      cudaMemcpy(CUSOLVER_DATA.h_d_w, CUSOLVER_DATA.d_d_w, N * sizeof(double),
                 cudaMemcpyDeviceToHost);

      if (CUSOLVER_DATA.verbose == 1) {
        printf(" DONE\n");
        fflush(stdout);
      }
    }

    if (LOOPmode == 0) {
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_FREE_PIN(CUSOLVER_DATA.h_f_R);
      } else {
        TESTING_FREE_PIN(CUSOLVER_DATA.h_d_R);
      }
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t5);

    if (CUSOLVER_DATA.verbose == 1) {
      printf("Write eigenvalues to file\n");
      fflush(stdout);
    }

    if (testmode == 1) {
      sprintf(fname, "test_eigenv.dat");
      if ((fp = fopen(fname, "w")) == NULL) {
        printf("ERROR: cannot create file \"%s\"\n", fname);
        exit(0);
      }
      if (CUSOLVER_DATA.comp_float == 1) {
        for (k = 0; k < N; k++) {
          fprintf(fp, "%5ld %20.8g  %20.8f  %20.8f\n", k,
                  CUSOLVER_DATA.h_f_w[N - k - 1],
                  CUSOLVER_DATA.h_f_w[N - k - 1] / CUSOLVER_DATA.h_f_w[N - 1],
                  SVDeps * SVDeps);
        }
      } else {
        for (k = 0; k < N; k++) {
          fprintf(fp, "%5ld %20.8g  %20.8f  %20.8f\n", k,
                  CUSOLVER_DATA.h_d_w[N - k - 1],
                  CUSOLVER_DATA.h_d_w[N - k - 1] / CUSOLVER_DATA.h_d_w[N - 1],
                  SVDeps * SVDeps);
        }
      }
      fclose(fp);
    }

    /// w1 values are the EIGENVALUES of AT A
    /// Note: w1 values are the SQUARE of the singular values of A

    // ****************************************************
    // STEP 5 :   Set eigenvalue limit
    // ****************************************************
    double egvlim;
    if (CUSOLVER_DATA.comp_float == 1) {
      egvlim = SVDeps * SVDeps * CUSOLVER_DATA.d_f_w[N - 1];
    } else {
      egvlim = SVDeps * SVDeps * CUSOLVER_DATA.h_d_w[N - 1];
    }

    MaxNBmodes1 = MaxNBmodes;
    if (MaxNBmodes1 > N) {
      MaxNBmodes1 = N;
    }
    if (MaxNBmodes1 > M) {
      MaxNBmodes1 = M;
    }
    mode = 0;

    if (CUSOLVER_DATA.comp_float == 1) {
      while ((mode < MaxNBmodes1) &&
             (CUSOLVER_DATA.d_f_w[N - mode - 1] > egvlim)) {
        mode++;
      }
    } else {
      while ((mode < MaxNBmodes1) &&
             (CUSOLVER_DATA.h_d_w[N - mode - 1] > egvlim)) {
        mode++;
      }
    }

    if (CUSOLVER_DATA.verbose == 1) {
      printf(
          "Keeping %ld modes  (SVDeps = %g -> %g, MaxNBmodes = %ld -> %ld)\n",
          mode, SVDeps, egvlim, MaxNBmodes, MaxNBmodes1);
      fflush(stdout);
    }

    if (testmode == 1) {
      fp = fopen("test_SVDmodes.log", "w");
      fprintf(fp, "%6ld %6ld\n", mode, MaxNBmodes1);
      fclose(fp);
    }
    MaxNBmodes1 = mode;
    // printf("Keeping %ld modes  (SVDeps = %g)\n", MaxNBmodes1, SVDeps);

    // ****************************************************
    // STEP 6 :   Write eigenvectors to VT matrix
    // ****************************************************

    // eigenvectors are in CUSOLVER_DATA.d_d_AtA (device), copy them to
    // CUSOLVER_DATA.h_d_AtA (host)

    if (CUSOLVER_DATA.comp_float == 1) {
      cublasGetMatrix(N, N, sizeof(float), CUSOLVER_DATA.d_f_AtA, N,
                      CUSOLVER_DATA.h_f_AtA, N);
    } else {
      cublasGetMatrix(N, N, sizeof(float), CUSOLVER_DATA.d_d_AtA, N,
                      CUSOLVER_DATA.h_d_AtA, N);
    }

    // copy eigenvectors from CUSOLVER_DATA.h_d_AtA to VT
    ID_VT = create_2Dimage_ID(ID_VTmatrix_name, N, N);

    if (CUSOLVER_DATA.comp_float == 1) {
      for (ii = 0; ii < N; ii++)
        for (jj = 0; jj < N; jj++) {
          data.image[ID_VT].array.F[jj * N + ii] =
              CUSOLVER_DATA.h_f_AtA[(N - ii - 1) * N + jj];
        }
    } else {
      for (ii = 0; ii < N; ii++)
        for (jj = 0; jj < N; jj++) {
          data.image[ID_VT].array.F[jj * N + ii] =
              CUSOLVER_DATA.h_d_AtA[(N - ii - 1) * N + jj];
        }
    }

    if (testmode == 1) {
      save_fits("mVT", "!test_mVT.fits");
    }

    // ****************************************************
    // STEP 7 :   Write eigenvectors/eigenvalue to CUSOLVER_DATA.h_d_VT1 if
    // eigenvalue > limit
    //          Copy to CUSOLVER_DATA.d_d_VT1
    // ****************************************************

    if (CUSOLVER_DATA.comp_float == 1) {
      for (ii = 0; ii < N; ii++)
        for (jj = 0; jj < N; jj++) {
          if (N - jj - 1 < MaxNBmodes1) {
            CUSOLVER_DATA.h_f_VT1[ii * N + jj] =
                CUSOLVER_DATA.h_f_AtA[jj * N + ii] / CUSOLVER_DATA.d_f_w[jj];
          } else {
            CUSOLVER_DATA.h_f_VT1[ii * N + jj] = 0.0;
          }
        }
      cublasSetMatrix(N, N, sizeof(float), CUSOLVER_DATA.h_f_VT1, N,
                      CUSOLVER_DATA.d_f_VT1, N);
    } else {
      for (ii = 0; ii < N; ii++)
        for (jj = 0; jj < N; jj++) {
          if (N - jj - 1 < MaxNBmodes1) {
            CUSOLVER_DATA.h_d_VT1[ii * N + jj] =
                CUSOLVER_DATA.h_d_AtA[jj * N + ii] / CUSOLVER_DATA.h_d_w[jj];
          } else {
            CUSOLVER_DATA.h_d_VT1[ii * N + jj] = 0.0;
          }
        }
      cublasSetMatrix(N, N, sizeof(double), CUSOLVER_DATA.h_d_VT1, N,
                      CUSOLVER_DATA.d_d_VT1, N);
    }

    if (LOOPmode == 0) {
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_FREE_CPU(CUSOLVER_DATA.h_f_VT1);
        TESTING_FREE_CPU(CUSOLVER_DATA.h_f_w);
      } else {
        TESTING_FREE_CPU(CUSOLVER_DATA.h_d_VT1);
        TESTING_FREE_CPU(CUSOLVER_DATA.h_d_w);
      }
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t6);

    // ****************************************************
    // STEP 8 :   Compute M2 = VT1 VT = (AT A)^-1
    // ****************************************************

    if (CUSOLVER_DATA.verbose == 1) {
      printf("compute M2 = VT1 VT\n");
      fflush(stdout);
    }

    if (CUSOLVER_DATA.comp_float == 1) {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> cublasSgemm ");
        fflush(stdout);
      }
      float alpha = 1, beta = 0;
      cublasSgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, N,
                  N, N, &alpha, CUSOLVER_DATA.d_f_VT1, N, CUSOLVER_DATA.d_f_AtA, N,
                  &beta, CUSOLVER_DATA.d_f_M2, N);

      if (CUSOLVER_DATA.verbose == 1) {
        printf("-> DONE\n");
        fflush(stdout);
      }
    } else {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> cublasDgemm ");
        fflush(stdout);
      }
      double alpha = 1, beta = 0;
      cublasDgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, N,
                  N, N, &alpha, CUSOLVER_DATA.d_d_VT1, N, CUSOLVER_DATA.d_d_AtA, N,
                  &beta, CUSOLVER_DATA.d_d_M2, N);

      if (CUSOLVER_DATA.verbose == 1) {
        printf("-> DONE\n");
        fflush(stdout);
      }
    }

    if (testmode == 1) {
      long ID_M2;

      ID_M2 = create_2Dimage_ID("mM2", N, N);
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_MALLOC_CPU(CUSOLVER_DATA.h_f_M2, float, N *N);
        cublasGetMatrix(N, N, sizeof(float), CUSOLVER_DATA.d_f_M2, N,
                        CUSOLVER_DATA.h_f_M2, N);

        for (ii = 0; ii < N; ii++)
          for (jj = 0; jj < N; jj++) {
            data.image[ID_M2].array.F[jj * N + ii] =
                CUSOLVER_DATA.h_f_M2[jj * N + ii];
          }
      } else {
        TESTING_MALLOC_CPU(CUSOLVER_DATA.h_d_M2, double, N *N);
        cublasGetMatrix(N, N, sizeof(double), CUSOLVER_DATA.d_d_M2, N,
                        CUSOLVER_DATA.h_d_M2, N);

        for (ii = 0; ii < N; ii++)
          for (jj = 0; jj < N; jj++) {
            data.image[ID_M2].array.F[jj * N + ii] =
                CUSOLVER_DATA.h_d_M2[jj * N + ii];
          }
      }

      save_fits("mM2", "!test_mM2.fits");

      //	magma_dsetmatrix( N, N, h_M2, N, d_M2, N, magmaqueue);
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_FREE_CPU(CUSOLVER_DATA.h_f_M2);
      } else {
        TESTING_FREE_CPU(CUSOLVER_DATA.h_d_M2);
      }
    }

    if (LOOPmode == 0) {
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_f_VT1);
      } else {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_d_VT1);
      }
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t7);

    // ****************************************************
    // STEP 9 :   Compute Ainv = M2 A = (AT A)^-1 A
    // ****************************************************

    // compute Ainv = M2 A
    if (CUSOLVER_DATA.loop_iter == 0) {
      dAinvMODE = 1;
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_MALLOC_DEV(CUSOLVER_DATA.d_f_Ainv, float, N *M);
      } else {
        TESTING_MALLOC_DEV(CUSOLVER_DATA.d_d_Ainv, double, N *M);
      }
    }

    if (CUSOLVER_DATA.comp_float == 1) {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> cublasSgemm ");
        fflush(stdout);
      }

      float alpha = 1, beta = 0;
      cublasSgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M,
                  N, N, &alpha, CUSOLVER_DATA.d_f_A, M, CUSOLVER_DATA.d_f_M2, N,
                  &beta, CUSOLVER_DATA.d_f_Ainv, M);

      if (CUSOLVER_DATA.verbose == 1) {
        printf("-> DONE\n");
        fflush(stdout);
      }
    } else {
      if (CUSOLVER_DATA.verbose == 1) {
        printf(" -> cublasDgemm ");
        fflush(stdout);
      }

      double alpha = 1, beta = 0;
      cublasDgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M,
                  N, N, &alpha, CUSOLVER_DATA.d_d_A, M, CUSOLVER_DATA.d_d_M2, N,
                  &beta, CUSOLVER_DATA.d_d_Ainv, M);

      if (CUSOLVER_DATA.verbose == 1) {
        printf("-> DONE\n");
        fflush(stdout);
      }
    }

    if (LOOPmode == 0) {
      if (CUSOLVER_DATA.comp_float == 1) {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_f_M2);
      } else {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_d_M2);
      }
    }

    // if(timing==1)
    clock_gettime(CLOCK_REALTIME, &t8);

    if (CUSOLVER_DATA.comp_float == 1) {
      cublasGetMatrix(M, N, sizeof(float), CUSOLVER_DATA.d_f_Ainv, M,
                      CUSOLVER_DATA.h_f_Ainv, M);
    } else {
      cublasGetMatrix(M, N, sizeof(double), CUSOLVER_DATA.d_d_Ainv, M,
                      CUSOLVER_DATA.h_d_Ainv, M);
    }

    for (int elem = 0; elem < 10; elem++) {
      printf("TEST CUSOLVER_DATA.h_f_Ainv[%2d] = %.16f\n", elem,
             CUSOLVER_DATA.h_f_Ainv[elem]);
    }

  }  // END STD MAGMA =================================================
  // End of QDWHPartial / MAGMA conditional

  //
  // At this point, pseudo-inverse is in CUSOLVER_DATA.h_d_Ainv or
  // CUSOLVER_DATA.h_f_Ainv
  //

  if (CUSOLVER_DATA.verbose == 1) {
    printf("END OF PSINV\n");
    fflush(stdout);
  }

  if (testmode == 1) {
    ID_Ainv = create_2Dimage_ID("mAinv", M, N);
    if (CUSOLVER_DATA.comp_float == 1) {
      {
        for (ii = 0; ii < M; ii++)
          for (jj = 0; jj < N; jj++) {
            data.image[ID_Ainv].array.F[jj * M + ii] =
                CUSOLVER_DATA.h_f_Ainv[jj * M + ii];
          }
      }
    } else {
      for (ii = 0; ii < M; ii++)
        for (jj = 0; jj < N; jj++) {
          data.image[ID_Ainv].array.F[jj * M + ii] =
              CUSOLVER_DATA.h_d_Ainv[jj * M + ii];
        }
    }

    save_fits("mAinv", "!test_mAinv.fits");
  }

  // if(timing==1)
  clock_gettime(CLOCK_REALTIME, &t9);

  if (CUSOLVER_DATA.loop_iter == 0) {
    if (data.image[ID_Rmatrix].md[0].naxis == 3) {
      arraysizetmp[0] = data.image[ID_Rmatrix].md[0].size[0];
      arraysizetmp[1] = data.image[ID_Rmatrix].md[0].size[1];
      arraysizetmp[2] = N;
    } else {
      arraysizetmp[0] = M;
      arraysizetmp[1] = N;
    }

    if (datatype == _DATATYPE_FLOAT) {
      ID_Cmatrix =
          create_image_ID(ID_Cmatrix_name, data.image[ID_Rmatrix].md[0].naxis,
                          arraysizetmp, _DATATYPE_FLOAT, 0, 0);
    } else {
      ID_Cmatrix =
          create_image_ID(ID_Cmatrix_name, data.image[ID_Rmatrix].md[0].naxis,
                          arraysizetmp, _DATATYPE_DOUBLE, 0, 0);
    }
  } else {
    ID_Cmatrix = image_ID(ID_Cmatrix_name);
  }

  clock_gettime(CLOCK_REALTIME, &t10);

  if (CUSOLVER_DATA.verbose == 1) {
    printf("write result\n");
    fflush(stdout);
  }

  if (datatype == _DATATYPE_FLOAT) {
    if (CUSOLVER_DATA.comp_float == 1) {
      memcpy(data.image[ID_Cmatrix].array.F, CUSOLVER_DATA.h_f_Ainv,
             sizeof(float) * M * N);
    } else {
      for (ii = 0; ii < M * N; ii++) {
        data.image[ID_Cmatrix].array.F[ii] = (float)CUSOLVER_DATA.h_d_Ainv[ii];
      }
    }
  } else {
    // sensors : M
    // actuator modes: N
    if (CUSOLVER_DATA.comp_float == 1) {
      for (ii = 0; ii < M * N; ii++) {
        data.image[ID_Cmatrix].array.D[ii] = CUSOLVER_DATA.h_f_Ainv[ii];
      }
    } else {
      memcpy(data.image[ID_Cmatrix].array.D, CUSOLVER_DATA.h_d_Ainv,
             sizeof(double) * M * N);
    }
  }
  /*
      }
  */

  // if(timing==1)
  clock_gettime(CLOCK_REALTIME, &t11);

  if (testmode == 1)  // compute product of Ainv with A
  {
    float alpha = 1, beta = 0;
    cublasSgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N,
                  N, M, &alpha, CUSOLVER_DATA.d_f_A, M, CUSOLVER_DATA.d_f_Ainv, M,
                  &beta, CUSOLVER_DATA.d_f_AtA, N);

    long ID_AinvA;

    ID_AinvA = create_2Dimage_ID("AinvA", N, N);

    // copy from GPU to CPU
    cublasGetMatrix(N, N, sizeof(float), CUSOLVER_DATA.d_f_AtA, N,
                    CUSOLVER_DATA.h_f_AtA, N);

    if (CUSOLVER_DATA.comp_float == 1) {
      memcpy(data.image[ID_AinvA].array.F, CUSOLVER_DATA.h_f_AtA,
             sizeof(float) * N * N);
    }

    save_fits("AinvA", "!test_AinvA.fits");
  }

  clock_gettime(CLOCK_REALTIME, &t12);

  ID_PFfmdat = image_ID("PFfmdat");
  if (ID_PFfmdat != -1) {
    printf("=============================================\n");
    printf("=========// OUTPUT M-M MULTIPLY //===========\n");
    printf("=============================================\n");

    printf("Transp(Ainv)     N x M   = %d x %d\n", N, M);
    printf("PFfmdat  M x K           = %d x %d\n",
           data.image[ID_PFfmdat].md[0].size[0],
           data.image[ID_PFfmdat].md[0].size[1]);
    long K = data.image[ID_PFfmdat].md[0].size[1];
    printf("K = %ld\n", K);

    float *d_f_PFfmdat;
    float *d_f_PF;
    float *h_f_PF;

    TESTING_MALLOC_DEV(d_f_PFfmdat, float, M *K);
    TESTING_MALLOC_DEV(d_f_PF, float, N *K);
    TESTING_MALLOC_CPU(h_f_PF, float, N *K);

    cublasSetMatrix(M, K, sizeof(float), data.image[ID_PFfmdat].array.F, M,
                    d_f_PFfmdat, M);

    float alpha = 1, beta = 0;
    cublasSgemm(CUSOLVER_DATA.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N,
                  K, M, &alpha, CUSOLVER_DATA.d_f_Ainv, M, d_f_PFfmdat, M,
                  &beta, d_f_PF, N);
    cublasGetMatrix(N, K, sizeof(float), d_f_PF, N,
                    h_f_PF, N);

    long ID_PF = create_2Dimage_ID("psinvPFmat", N, K);
    list_image_ID();
    memcpy(data.image[ID_PF].array.F, h_f_PF,
           sizeof(float) * N * K);
    save_fits("psinvPFmat", "!psinvPFmat.fits");

    TESTING_FREE_DEV(d_f_PFfmdat);
    TESTING_FREE_DEV(d_f_PF);
    TESTING_FREE_CPU(h_f_PF);
  }

  clock_gettime(CLOCK_REALTIME, &t13);

  if (LOOPmode ==
      0)  /// if pseudo-inverse is only computed once, these arrays can be freed
  {
    if (CUSOLVER_DATA.comp_float == 1) {
      TESTING_FREE_CPU(CUSOLVER_DATA.h_f_A);
    } else {
      TESTING_FREE_CPU(CUSOLVER_DATA.h_d_A);
    }
  }

  if (LOOPmode == 0) {
    if (CUSOLVER_DATA.comp_float == 1) {
      TESTING_FREE_DEV(CUSOLVER_DATA.d_f_A);

      if (dAinvMODE == 1) {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_f_Ainv);
      }

      TESTING_FREE_CPU(CUSOLVER_DATA.h_f_Ainv);
      TESTING_FREE_DEV(CUSOLVER_DATA.d_f_AtA);
      TESTING_FREE_CPU(CUSOLVER_DATA.h_f_AtA);
    } else {
      TESTING_FREE_DEV(CUSOLVER_DATA.d_d_A);

      if (dAinvMODE == 1) {
        TESTING_FREE_DEV(CUSOLVER_DATA.d_d_Ainv);
      }

      TESTING_FREE_CPU(CUSOLVER_DATA.h_d_Ainv);
      TESTING_FREE_DEV(CUSOLVER_DATA.d_d_AtA);
      TESTING_FREE_CPU(CUSOLVER_DATA.h_d_AtA);
    }
  }

  if (LOOPmode == 0) {
    // magma_queue_destroy(magmaqueue);
    // magma_finalize();  //  finalize  Magma
  }

  free(arraysizetmp);

  // if(timing==1)
  //{
  t01d = timespec_diff_double(t0, t1);

  t12d = timespec_diff_double(t1, t2);
  t23d = timespec_diff_double(t2, t3);
  t34d = timespec_diff_double(t3, t4);
  t45d = timespec_diff_double(t4, t5);
  t56d = timespec_diff_double(t5, t6);
  t67d = timespec_diff_double(t6, t7);
  t78d = timespec_diff_double(t7, t8);
  t89d = timespec_diff_double(t8, t9);
  t910d = timespec_diff_double(t9, t10);
  t1011d = timespec_diff_double(t10, t11);
  t1112d = timespec_diff_double(t11, t12);
  t1213d = timespec_diff_double(t12, t13);
  t013d = timespec_diff_double(t0, t13);

  if (CUSOLVER_DATA.verbose == 1) {
    printf("%6ld  Timing info: \n", CUSOLVER_DATA.loop_iter);
    printf("  0-1	[setup]                           %12.3f ms\n",
           t01d * 1000.0);
    printf("  1-2	[copy input to GPU]               %12.3f ms\n",
           t12d * 1000.0);

    printf("  2-3	[compute trans(A) x A]            %12.3f ms\n",
           t23d * 1000.0);
    printf("  3-4	[setup]                           %12.3f ms\n",
           t34d * 1000.0);
    printf("  4-5	[Compute eigenvalues]             %12.3f ms\n",
           t45d * 1000.0);
    printf("  5-6	[Select eigenvalues]              %12.3f ms\n",
           t56d * 1000.0);
    printf("  6-7	[Compute M2]                      %12.3f ms\n",
           t67d * 1000.0);
    printf("  7-8	[Compute Ainv]                    %12.3f ms\n",
           t78d * 1000.0);
    printf("  8-9	[Get Ainv from GPU]               %12.3f ms\n",
           t89d * 1000.0);

    printf("  9-10	[output setup]                    %12.3f ms\n",
           t910d * 1000.0);
    printf("  10-11	[Write output array]              %12.3f ms\n",
           t1011d * 1000.0);
    printf("  11-12	[Test output]                     %12.3f ms\n",
           t1112d * 1000.0);
    printf("  12-13	[Optional gemm]                   %12.3f ms\n",
           t1213d * 1000.0);
    printf("\n");
    printf(" TOTAL 0-13     %12.3f ms\n", t013d * 1000.0);
    fflush(stdout);
  }
  //}

  if (CUSOLVER_DATA.verbose == 1) {
    printf("\n\n");
    fflush(stdout);
  }

  if (LOOPmode == 1) {
    CUSOLVER_DATA.loop_iter++;
  }

  return (ID_Cmatrix);
}

#endif
