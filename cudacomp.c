/**
 * @file    cudacomp.c
 * @brief   CUDA functions wrapper
 *
 * Also uses MAGMA library
 *
 *
 *
 */


#define MODULE_SHORTNAME_DEFAULT "cuda"
#define MODULE_DESCRIPTION       "CUDA wrapper"







/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

// include sem_timedwait
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE	200809L
#endif


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <signal.h>


#include <semaphore.h>


#include <time.h>





#include <sys/types.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>


#ifdef HAVE_CUDA

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_types.h>
#include <pthread.h>
#include <cusolverDn.h>

#include "cudacomp_types.h"

#endif


#ifdef HAVE_MAGMA
//#include "magma.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#endif



#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_tools/COREMOD_tools.h"


//#include "cudacomp/cudacomp.h"

#include "cudacomp_types.h"

#include "Coeff2Map_Loop.h"
#include "cudacomp_MVMextractModesLoop.h"
#include "cudacompinit.h"
#include "cudacomptest.h"
#include "magma_compute_SVDpseudoInverse.h"
#include "magma_compute_SVDpseudoInverse_SVD.h"
#include "MatMatMult_testPseudoInverse.h"


#include "linopt_imtools/linopt_imtools.h" // for testing



/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

# ifdef _OPENMP
# include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
# endif

static inline int r32up(int x)
{
    return ((x + 31) / 32) * 32;
}



/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */



extern pid_t CLIPID;




imageID IDtimerinit = 0;
imageID IDtiming = -1; // index to image where timing should be written





#ifdef HAVE_CUDA
int cuda_deviceCount;
GPUMATMULTCONF gpumatmultconf[20]; // supports up to 20 configurations per process
float cublasSgemv_alpha = 1.0;
float cublasSgemv_beta  = 0.0;
#endif




// MAGMA global variables

#ifdef HAVE_MAGMA
int INIT_MAGMA = 0;

// queue for default magma device
magma_queue_t   magmaqueue;
#endif







/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(cudacomp)


static void __attribute__((constructor)) libinit_cudacomp_printinfo()
{
#ifdef HAVE_CUDA
    printf("[CUDA]");
#endif

#ifdef HAVE_MAGMA
    printf("[MAGMA]");
#endif
}









static errno_t init_module_CLI()
{
    long i;

#ifdef HAVE_CUDA
    //    printf("HAVE_CUDA defined\n");
    for(i = 0; i < 20; i++)
    {
        gpumatmultconf[i].init = 0;
        gpumatmultconf[i].alloc = 0;
    }


	cudacompinit_addCLIcmd();
	cudacomptest_addCLIcmd();
	MatMatMult_testPseudoInverse_addCLIcmd();
	magma_compute_SVDpseudoInverse_addCLIcmd();
	magma_compute_SVDpseudoInverse_SVD_addCLIcmd();
	Coeff2Map_Loop_addCLIcmd();
	cudacomp_MVMextractModesLoop_addCLIcmd();

#endif
    // add atexit functions here

    return RETURN_SUCCESS;
}










#ifdef HAVE_CUDA






errno_t CUDACOMP_printGPUMATMULTCONF(int index)
{
    printf("\n");
    printf("============= GPUMATMULTCONF %d ======================\n", index);
    printf(" init              = %20d\n", (int) gpumatmultconf[index].init);
    printf(" refWFSinit        = %p\n", (void *) gpumatmultconf[index].refWFSinit);

    if(gpumatmultconf[index].refWFSinit != NULL)
    {
        printf("     refWFSinit[0]     = %20d\n",
               (int) gpumatmultconf[index].refWFSinit[0]);
    }

    printf(" alloc             = %20d\n", (int) gpumatmultconf[index].alloc);
    printf(" CM_ID             = %20ld\n", gpumatmultconf[index].CM_ID);
    printf(" CM_cnt            = %20ld\n", gpumatmultconf[index].CM_cnt);
    printf(" timerID           = %20ld\n", gpumatmultconf[index].timerID);
    printf(" M                 = %20d\n", (int) gpumatmultconf[index].M);
    printf(" N                 = %20d\n", (int) gpumatmultconf[index].N);

    /// synchronization
    printf(" sem               = %20d\n", (int) gpumatmultconf[index].sem);
    printf(" gpuinit           = %20d\n", (int) gpumatmultconf[index].gpuinit);

    /// one semaphore per thread
    /*
        sem_t **semptr1;
        sem_t **semptr2;
        sem_t **semptr3;
        sem_t **semptr4;
        sem_t **semptr5;
    */

    printf(" cMat              = %20p\n", (void *) gpumatmultconf[index].cMat);
    printf(" cMat_part         = %20p\n", (void *) gpumatmultconf[index].cMat_part);
    printf(" wfsVec            = %20p\n", (void *) gpumatmultconf[index].wfsVec);
    printf(" wfsVec_part       = %20p\n",
           (void *) gpumatmultconf[index].wfsVec_part);
    printf(" wfsRef            = %20p\n", (void *) gpumatmultconf[index].wfsRef);
    printf(" wfsRef_part       = %20p\n",
           (void *) gpumatmultconf[index].wfsRef_part);
    printf(" dmVec             = %20p\n", (void *) gpumatmultconf[index].dmVec);
    printf(" dmVecTMP          = %20p\n", (void *) gpumatmultconf[index].dmVecTMP);
    printf(" dmVec_part        = %20p\n",
           (void *) gpumatmultconf[index].dmVec_part);
    printf(" dmRef_part        = %20p\n",
           (void *) gpumatmultconf[index].dmRef_part);



    printf(" d_cMat            = %20p\n", (void *) gpumatmultconf[index].d_cMat);
    printf(" d_wfsVec          = %20p\n", (void *) gpumatmultconf[index].d_wfsVec);
    printf(" d_dmVec           = %20p\n", (void *) gpumatmultconf[index].d_dmVec);
    printf(" d_wfsRef          = %20p\n", (void *) gpumatmultconf[index].d_wfsRef);
    printf(" d_dmRef           = %20p\n", (void *) gpumatmultconf[index].d_dmRef);


    // threads
    printf(" thdata            = %20p\n", (void *) gpumatmultconf[index].thdata);
    printf(" threadarray       = %20p\n",
           (void *) gpumatmultconf[index].threadarray);
    printf(" NBstreams         = %20d\n", (int) gpumatmultconf[index].NBstreams);
    printf(" stream            = %20p\n", (void *) gpumatmultconf[index].stream);
    printf(" handle            = %20p\n", (void *) gpumatmultconf[index].handle);


    printf(" Nsize             = %20p\n", (void *) gpumatmultconf[index].Nsize);
    printf(" Noffset           = %20p\n", (void *) gpumatmultconf[index].Noffset);
    printf(" GPUdevice         = %20p\n", (void *) gpumatmultconf[index].GPUdevice);

    printf(" orientation       = %20d\n", (int) gpumatmultconf[index].orientation);

    printf("======================================================\n");
    printf("\n");

    return RETURN_SUCCESS;
}






/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 2. LOW-LEVEL MATRIX VECTOR MULTIPLICATION FUNCTIONS                                             */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



void matrixMulCPU(
    float *cMat,
    float *wfsVec,
    float *dmVec,
    int    M,
    int    N
)
{
    printf("Conventional mat mult %d %d\n", M, N);
    for(int m = 0; m < M; m++)
    {
        dmVec[m] = 0.0;
        for(int n = 0; n < N; n++)
        {
            int index = m * N + n;
            dmVec[m] += cMat[index] * wfsVec[n];
        }
        //cMat[n*M+m]*wfsVec[n];
    }

    printf("cMat  : ");
    for(int i = 0; i < 5; i++)
    {
        printf("%f ", cMat[i]);
    }
    printf(" ... ");
    for(int i = N * M - 5; i < N * M; i++)
    {
        printf("%f ", cMat[i]);
    }
    printf("\n");

    printf("wfsVec: ");
    for(int n = 0; n < 5; n++)
    {
        printf("%f ", wfsVec[n]);
    }
    printf(" ... ");
    for(int n = N - 5; n < N; n++)
    {
        printf("%f ", wfsVec[n]);
    }
    printf("\n");

}








// extract mode coefficients from data stream
/*
int CUDACOMP_createModesLoop(const char *DMmodeval_stream, const char *DMmodes, const char *DMact_stream, int GPUindex)
{
    long ID_DMmodeval;
    long ID_DMmodes;
    long ID_DMact;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    struct cudaDeviceProp deviceProp;
    int m, n;
    int k;
    long *arraytmp;

    float *d_DMmodes = NULL; // linear memory of GPU
    float *d_DMact = NULL;
    float *d_modeval = NULL;

    float alpha = 1.0;
    float beta = 0.0;
    int loopOK;
    struct timespec ts;
    long iter;
    long long cnt = -1;
    long scnt;
    int semval;
    int semr;
    long ii, kk;

    long NBmodes;

    float *normcoeff;



    ID_DMact = image_ID(DMact_stream);
    m = data.image[ID_DMact].md[0].size[0]*data.image[ID_DMact].md[0].size[1];

    ID_DMmodes = image_ID(DMmodes);
    n = data.image[ID_DMmodes].md[0].size[2];
    NBmodes = n;
    normcoeff = (float*) malloc(sizeof(float)*NBmodes);

    for(kk=0;kk<NBmodes;kk++)
        {
            normcoeff[kk] = 0.0;
            for(ii=0;ii<m;ii++)
                normcoeff[kk] += data.image[ID_DMmodes].array.F[kk*m+ii]*data.image[ID_DMmodes].array.F[kk*m+ii];
            for(ii=0;ii<m;ii++)
                data.image[ID_DMmodes].array.F[kk*m+ii] /= normcoeff[kk];
        }

    //NBmodes = 3;

    arraytmp = (long*) malloc(sizeof(long)*2);
    arraytmp[0] = NBmodes;
    arraytmp[1] = 1;
    ID_modeval = create_image_ID(DMmodes_val, 2, arraytmp, _DATATYPE_FLOAT, 1, 0);
    free(arraytmp);
    COREMOD_MEMORY_image_set_createsem(DMmodes_val, 2);


    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%d devices found\n", cuda_deviceCount);
    fflush(stdout);
    printf("\n");
    for (k = 0; k < cuda_deviceCount; ++k) {
        cudaGetDeviceProperties(&deviceProp, k);
        printf("Device %d [ %20s ]  has compute capability %d.%d.\n",
               k, deviceProp.name, deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("\n");
    }


    if(GPUindex<cuda_deviceCount)
        cudaSetDevice(GPUindex);
    else
    {
        printf("Invalid Device : %d / %d\n", GPUindex, cuda_deviceCount);
        exit(0);
    }


    printf("Create cublas handle ...");
    fflush(stdout);
    cublas_status = cublasCreate(&cublasH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    printf(" done\n");
    fflush(stdout);


    // load DMmodes to GPU
    cudaStat = cudaMalloc((void**)&d_DMmodes, sizeof(float)*m*NBmodes);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_DMmodes returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaStat = cudaMemcpy(d_DMmodes, data.image[ID_DMmodes].array.F, sizeof(float)*m*NBmodes, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }


    // create d_DMact
    cudaStat = cudaMalloc((void**)&d_DMact, sizeof(float)*m);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_DMact returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }

    // create d_modeval
    cudaStat = cudaMalloc((void**)&d_modeval, sizeof(float)*NBmodes);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_modeval returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }


    if (sigaction(SIGINT, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGTERM, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGBUS, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGSEGV, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGABRT, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGHUP, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGPIPE, &data.sigact, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }


    loopOK = 1;
    iter = 0;

    while(loopOK == 1)
    {
        if(data.image[ID_DMact].md[0].sem==0)
        {
            while(data.image[ID_DMact].md[0].cnt0==cnt) // test if new frame exists
                usleep(5);
            cnt = data.image[ID_DMact].md[0].cnt0;
            semr = 0;
        }
        else
        {
            if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;
            semr = sem_timedwait(data.image[ID_DMact].semptr[0], &ts);

            if(iter == 0)
            {
                printf("driving semaphore to zero ... ");
                fflush(stdout);
                sem_getvalue(data.image[ID_DMact].semptr[0], &semval);
                for(scnt=0; scnt<semval; scnt++)
                    sem_trywait(data.image[ID_DMact].semptr[0]);
                printf("done\n");
                fflush(stdout);
            }
        }

        if(semr==0)
        {

            // load DMact to GPU
            cudaStat = cudaMemcpy(d_DMact, data.image[ID_DMact].array.F, sizeof(float)*m, cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess)
            {
                printf("cudaMemcpy returned error code %d, line(%d)\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }

            // compute
            cublas_status = cublasSgemv(cublasH, CUBLAS_OP_T, m, NBmodes, &alpha, d_DMmodes, m, d_DMact, 1, &beta, d_modeval, 1);
            if (cudaStat != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n", stat, __LINE__);
                if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                if(stat == CUBLAS_STATUS_INVALID_VALUE)
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                if(stat == CUBLAS_STATUS_ARCH_MISMATCH)
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                if(stat == CUBLAS_STATUS_EXECUTION_FAILED)
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                exit(EXIT_FAILURE);
            }

            // copy result
            data.image[ID_modeval].md[0].write = 1;
            cudaStat = cudaMemcpy(data.image[ID_modeval].array.F, d_modeval, sizeof(float)*NBmodes, cudaMemcpyDeviceToHost);
            sem_getvalue(data.image[ID_modeval].semptr[0], &semval);
            if(semval<SEMAPHORE_MAXVAL)
                sem_post(data.image[ID_modeval].semptr[0]);
            sem_getvalue(data.image[ID_modeval].semptr[1], &semval);
            if(semval<SEMAPHORE_MAXVAL)
                sem_post(data.image[ID_modeval].semptr[1]);
            data.image[ID_modeval].md[0].cnt0++;
            data.image[ID_modeval].md[0].write = 0;
        }

        if((data.signal_INT == 1)||(data.signal_TERM == 1)||(data.signal_ABRT==1)||(data.signal_BUS==1)||(data.signal_SEGV==1)||(data.signal_HUP==1)||(data.signal_PIPE==1))
            loopOK = 0;

        iter++;
    }


    cudaFree(d_DMmodes);
    cudaFree(d_DMact);
    cudaFree(d_modeval);

    if (cublasH ) cublasDestroy(cublasH);

    free(normcoeff);

    return(0);
}

*/


























#endif










