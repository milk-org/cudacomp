/** @file magma_compute_SVDpseudoInverse.h
 */

#ifndef MAGMA_COMPUTE_SVDPSEUDOINVERSE_H
#define MAGMA_COMPUTE_SVDPSEUDOINVERSE_H

#if defined(HAVE_CUDA) && defined(HAVE_MAGMA)

errno_t magma_compute_SVDpseudoInverse_addCLIcmd();


int CUDACOMP_magma_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name,
    const char *ID_Cmatrix_name,
    double      SVDeps,
    long        MaxNBmodes,
    const char *ID_VTmatrix_name,
    int         LOOPmode,
    int         PSINV_MODE,
    __attribute__((unused)) double      qdwh_s,
    __attribute__((unused)) float       qdwh_tol,
    int 		testmode
);

#endif  // defined(HAVE_CUDA) && defined(HAVE_MAGMA)

#endif  // MAGMA_COMPUTE_SVDPSEUDOINVERSE_H
