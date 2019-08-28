/**
 * @file    cudacomp_MVMextractModesLoop.c
 * @brief   CUDA functions wrapper
 * 
 * Requires CUDA library
 *  
 */


// uncomment for test print statements to stdout
#define _PRINT_TEST


/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

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

#endif




#include "CommandLineInterface/CLIcore.h"
#include "00CORE/00CORE.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "info/info.h"
#include "cudacomp/cudacomp.h"

#include "linopt_imtools/linopt_imtools.h" // for testing







/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */








        
    
    








#ifdef HAVE_CUDA



//
// manages configuration parameters
// initializes configuration parameters structure
//

errno_t CUDACOMP_MVMextractModesLoop_FPCONF(
    char *fpsname,
    uint32_t CMDmode
) {

    // ===========================
    // SETUP FPS
    // ===========================
    uint16_t loopstatus;
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fpsname, CMDmode, &loopstatus);


    // ===========================
    // ALLOCATE FPS ENTRIES
    // ===========================


    void *pNull = NULL;

    uint64_t FPFLAG;
    FPFLAG = FPFLAG_DEFAULT_INPUT;
    FPFLAG &= FPFLAG_WRITECONF;
    FPFLAG &= ~FPFLAG_WRITERUN;



	long GPUindex_default[4] = { 0, 0, 9, 0 };
    long fp_GPUindex        = function_parameter_add_entry(&fps, ".GPUindex", "GPU index",
                              FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, &GPUindex_default);


    long fp_streamname_in          = function_parameter_add_entry(&fps, ".sname_in",  "input stream vector",
                                     FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, pNull);

    long fp_streamname_modes       = function_parameter_add_entry(&fps, ".sname_modes",  "input modes matrix",
                                     FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, pNull);

	FPFLAG = FPFLAG_DEFAULT_INPUT_STREAM;
	FPFLAG &= ~FPFLAG_STREAM_RUN_REQUIRED;
    long fp_streamname_intot       = function_parameter_add_entry(&fps, ".option.sname_intot",  "optional input normalization stream",
                                     FPTYPE_STREAMNAME, FPFLAG, pNull);

    long fp_streamname_refin       = function_parameter_add_entry(&fps, ".option.sname_refin",  "optional input reference to be subtracted stream",
                                     FPTYPE_STREAMNAME, FPFLAG, pNull);

    long fp_streamname_refout      = function_parameter_add_entry(&fps, ".option.sname_refout",  "optional output reference to be subtracted stream",
                                     FPTYPE_STREAMNAME, FPFLAG, pNull);

    long fp_stream_outmodesval     = function_parameter_add_entry(&fps, ".sname_outmodesval", "output mode coefficients stream",
                                     FPTYPE_STREAMNAME, FPFLAG, pNull);

    long fp_outinit                = function_parameter_add_entry(&fps, ".outinit", "output stream init mode",
                                     FPTYPE_ONOFF, FPFLAG, pNull);



    long fp_PROCESS         = function_parameter_add_entry(&fps, ".option.PROCESS", "1 if processing",
                              FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);

    long fp_TRACEMODE       = function_parameter_add_entry(&fps, ".option.TRACEMODE", "1 if writing trace",
                              FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);

    long fp_MODENORM        = function_parameter_add_entry(&fps, ".option.MODENORM", "1 if input modes should be normalized",
                              FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);

    long fp_insem           = function_parameter_add_entry(&fps, ".option.insem", "input semaphore index",
                              FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, pNull);

    long fp_axmode          = function_parameter_add_entry(&fps, ".option.axmode", "0 for normal mode extraction, 1 for expansion",
                              FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, pNull);

    long fp_twait           = function_parameter_add_entry(&fps, ".option.twait", "if >0, insert time wait [us] at each iteration",
                              FPTYPE_INT64, FPFLAG_DEFAULT_INPUT|FPFLAG_WRITERUN, pNull);

    long fp_semwarn         = function_parameter_add_entry(&fps, ".option.semwarn", "issue warning when input stream semaphore >1",
                              FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);


    if(loopstatus == 0) { // stop fps
        return RETURN_SUCCESS;
    }


    // =====================================
    // PARAMETER LOGIC AND UPDATE LOOP
    // =====================================


    while(loopstatus == 1) {
        if(function_parameter_FPCONFloopstep(&fps, CMDmode, &loopstatus) == 1) { // Apply logic if update is needed
            // here goes the logic
            
            
            

            functionparameter_CheckParametersAll(&fps);  // check all parameter values
        }
    }
    function_parameter_FPCONFexit(&fps);


    return RETURN_SUCCESS;
}







errno_t __attribute__((hot)) CUDACOMP_MVMextractModesLoop_RUN(
    char *fpsname
) {
    long IDmodes;
    long ID;
    long ID_modeval;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    struct cudaDeviceProp deviceProp;
    int m, n;
    int k;
    uint32_t *arraytmp;

    float *d_modes = NULL; // linear memory of GPU
    float *d_in = NULL;
    float *d_modeval = NULL;

    float alpha = 1.0;
    float beta = 0.0;
    struct timespec ts;
    long loopcnt;
    long long cnt = -1;
    long scnt;
    int semval;
    int semr;
    long ii, jj, kk;

    long NBmodes;
    float *normcoeff;

    long IDoutact;
    uint32_t *sizearraytmp;

    long ID_modeval_mult;
    int imOK;


    char traceim_name[200];
    long TRACEsize = 2000;
    long TRACEindex = 0;
    long IDtrace;


    int NBaveSTEP = 10; // each step is 2x longer average than previous step
    double stepcoeff;
    double stepcoeff0 = 0.3;
    char process_ave_name[200];
    char process_rms_name[200];
    long IDprocave;
    long IDprocrms;
    long step;

    long semnb;
    double tmpv;

    int INNORMMODE = 0; // 1 if input normalized

    float *modevalarray;
    float *modevalarrayref;

    int initref = 0; // 1 when reference has been processed
    int BETAMODE = 0;
    long IDrefout;

    long refindex;
    long twait1;
    struct timespec t0;

    struct timespec t00;
    struct timespec t01;
    struct timespec t02;
    struct timespec t03;
    struct timespec t04;
    struct timespec t05;
    struct timespec t06;


    struct timespec t1;

    int MODEVALCOMPUTE = 1; // 1 if compute, 0 if import


    int RT_priority = 91; //any number from 0-99



    // ===========================
    // CONNECT TO FPS
    // ===========================
    FUNCTION_PARAMETER_STRUCT fps;
    if(function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_RUN) == -1) {
        printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", fpsname);
        return RETURN_FAILURE;
    }

    // ===============================
    // GET FUNCTION PARAMETER VALUES
    // ===============================

	char in_stream[200];
    strncpy(in_stream,  functionparameter_GetParamPtr_STRING(&fps, ".sname_in"),  FUNCTION_PARAMETER_STRMAXLEN);

	char IDmodes_name[200];
    strncpy(IDmodes_name,  functionparameter_GetParamPtr_STRING(&fps, ".sname_modes"),  FUNCTION_PARAMETER_STRMAXLEN);


	char intot_stream[200];
    strncpy(intot_stream,  functionparameter_GetParamPtr_STRING(&fps, ".option.sname_intot"),  FUNCTION_PARAMETER_STRMAXLEN);

	char IDrefin_name[200];
    strncpy(IDrefin_name,  functionparameter_GetParamPtr_STRING(&fps, ".option.sname_refin"),  FUNCTION_PARAMETER_STRMAXLEN);

	char IDrefout_name[200];
    strncpy(IDrefout_name,  functionparameter_GetParamPtr_STRING(&fps, ".option.sname_refout"),  FUNCTION_PARAMETER_STRMAXLEN);

	char IDmodes_val_name[200];
    strncpy(IDmodes_val_name,  functionparameter_GetParamPtr_STRING(&fps, ".sname_outmodesval"),  FUNCTION_PARAMETER_STRMAXLEN);

	int outinit = functionparameter_GetParamValue_ONOFF(&fps, ".outinit");


	int GPUindex    = functionparameter_GetParamValue_INT64(&fps, ".GPUindex");
	int PROCESS     = functionparameter_GetParamValue_ONOFF(&fps, ".option.PROCESS");
	int TRACEMODE   = functionparameter_GetParamValue_ONOFF(&fps, ".option.TRACEMODE");
	int MODENORM    = functionparameter_GetParamValue_ONOFF(&fps, ".option.MODENORM");
	int insem       = functionparameter_GetParamValue_INT64(&fps, ".option.insem");
	int axmode      = functionparameter_GetParamValue_INT64(&fps, ".option.axmode");
	long *twait     = functionparameter_GetParamPtr_INT64(&fps, ".option.twait");
	int semwarn     = functionparameter_GetParamValue_ONOFF(&fps, ".option.semwarn");
	


    // ===============================
    // Review input parameters
    // ===============================

    printf("\n");
    printf("in_stream        : %16s  input stream\n", in_stream);
    printf("intot_stream     : %16s  [optional] input normalization stream\n", intot_stream);
    printf("IDmodes_name     : %16s  Modes\n", IDmodes_name);
    printf("IDrefin_name     : %16s  [optional] input reference  - to be subtracted\n", IDrefin_name);
    printf("IDrefout_name    : %16s  [optional] output reference - to be added\n", IDrefout_name);
    printf("IDmodes_val_name : %16s  ouput stream\n", IDmodes_val_name);
    
    printf("GPUindex         : %16d  GPU index\n", GPUindex);
    printf("PROCESS          : %16d  1 if postprocessing\n", PROCESS);
    printf("TRACEMODE        : %16d  1 if writing trace\n", TRACEMODE);
    printf("MODENORM         : %16d  1 if input modes should be normalized\n", MODENORM);
    printf("insem            : %16d  input semaphore index\n", insem);
    printf("axmode           : %16d  0 for normal mode extraction, 1 for expansion\n", axmode);
    printf("twait            : %16ld  if >0, insert time wait [us] at each iteration\n", *twait);
    printf("semwarn          : %16d  1 if warning when input stream semaphore >1\n", semwarn);
    printf("\n");



    // ===========================
    // processinfo support 
    // ===========================

    char pinfoname[200];
    sprintf(pinfoname, "cudaMVMextract-%s", in_stream);

    char pinfodescr[200];
    sprintf(pinfodescr, "%s->%s", in_stream, IDmodes_val_name);

    char pinfomsg[200];
    sprintf(pinfomsg, "Setup");

    PROCESSINFO *processinfo;
    processinfo = processinfo_setup(
                      pinfoname,             // short name for the processinfo instance, no spaces, no dot, name should be human-readable
                      pinfodescr,    // description
                      pinfomsg,  // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );

    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority = RT_priority;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore

    int loopOK = 1;









    // ===========================
    // INITIALIZATIONS 
    // ===========================


    // CONNECT TO INPUT STREAM
    long IDin;
    IDin = image_ID(in_stream);

    // ERROR HANDLING
    if(IDin == -1) {
        struct timespec errtime;
        struct tm *errtm;

        clock_gettime(CLOCK_REALTIME, &errtime);
        errtm = gmtime(&errtime.tv_sec);

        fprintf(stderr,
                "%02d:%02d:%02d.%09ld  ERROR [%s %s %d] Input stream %s does not exist, cannot proceed\n",
                errtm->tm_hour,
                errtm->tm_min,
                errtm->tm_sec,
                errtime.tv_nsec,
                __FILE__,
                __FUNCTION__,
                __LINE__,
                in_stream);
        return 1;
    }


    m = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1];
    COREMOD_MEMORY_image_set_createsem(in_stream, 10);


    // CONNECT TO TOTAL FLUX STREAM
    long IDintot;
    IDintot = image_ID(intot_stream);
    if(IDintot == -1) {
        INNORMMODE = 0;
        IDintot = create_2Dimage_ID("intot_tmp", 1, 1);
        data.image[IDintot].array.F[0] = 1.0;
    } else {
        INNORMMODE = 1;
    }




    // CONNECT TO WFS REFERENCE STREAM
    long IDref;
    IDref = image_ID(IDrefin_name);
    if(IDref == -1) {
        IDref = create_2Dimage_ID("_tmprefin", data.image[IDin].md[0].size[0], data.image[IDin].md[0].size[1]);
        for(ii = 0; ii < data.image[IDin].md[0].size[0]*data.image[IDin].md[0].size[1]; ii++) {
            data.image[IDref].array.F[ii] = 0.0;
        }
    }



    if(axmode == 0) {
        IDmodes = image_ID(IDmodes_name);
        n = data.image[IDmodes].md[0].size[2];
        NBmodes = n;
    } else {
        ID = image_ID(IDmodes_name);
        printf("Modes: ID = %ld   %s\n", ID, IDmodes_name);
        fflush(stdout);

        NBmodes = data.image[ID].md[0].size[0] * data.image[ID].md[0].size[1];
        n = NBmodes;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);

        printf("creating _tmpmodes  %ld %ld %ld\n", (long) data.image[IDin].md[0].size[0], (long) data.image[IDin].md[0].size[1], NBmodes);
        fflush(stdout);


        IDmodes = create_3Dimage_ID("_tmpmodes", data.image[IDin].md[0].size[0], data.image[IDin].md[0].size[1], NBmodes);

        for(ii = 0; ii < data.image[IDin].md[0].size[0]; ii++)
            for(jj = 0; jj < data.image[IDin].md[0].size[1]; jj++) {
                for(kk = 0; kk < NBmodes; kk++) {
                    data.image[IDmodes].array.F[kk * data.image[IDin].md[0].size[0]*data.image[IDin].md[0].size[1] + jj * data.image[IDin].md[0].size[0] + ii] = data.image[ID].array.F[NBmodes * (jj * data.image[IDin].md[0].size[0] + ii) + kk];
                }
            }
                
        //save_fits("_tmpmodes", "!_test_tmpmodes.fits");
    }



    normcoeff = (float *) malloc(sizeof(float) * NBmodes);

    if(MODENORM == 1) {
        for(k = 0; k < NBmodes; k++) {
            normcoeff[k] = 0.0;
            for(ii = 0; ii < m; ii++) {
                normcoeff[k] += data.image[IDmodes].array.F[k * m + ii] * data.image[IDmodes].array.F[k * m + ii];
            }
        }
    } else
        for(k = 0; k < NBmodes; k++) {
            normcoeff[k] = 1.0;
        }





    modevalarray = (float *) malloc(sizeof(float) * n);
    modevalarrayref = (float *) malloc(sizeof(float) * n);


    arraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

    IDrefout = image_ID(IDrefout_name);
    if(IDrefout == -1) {
        arraytmp[0] = NBmodes;
        arraytmp[1] = 1;
    } else {
        arraytmp[0] = data.image[IDrefout].md[0].size[0];
        arraytmp[1] = data.image[IDrefout].md[0].size[1];
    }



	// CONNNECT TO OUTPUT STREAM

    ID_modeval = image_ID(IDmodes_val_name);
    if(ID_modeval == -1) { // CREATE IT
        ID_modeval = create_image_ID(IDmodes_val_name, 2, arraytmp, _DATATYPE_FLOAT, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDmodes_val_name, 10);
        MODEVALCOMPUTE = 1;
    } else { // USE STREAM, DO NOT COMPUTE IT
        printf("======== Using pre-existing stream %s, insem = %d\n", IDmodes_val_name, insem);
        fflush(stdout);
        
        if( outinit == 0 )
			MODEVALCOMPUTE = 0;
		else
			MODEVALCOMPUTE = 1;
        
        // drive semaphore to zero
        while(sem_trywait(data.image[ID_modeval].semptr[insem]) == 0) {
            printf("WARNING %s %d  : sem_trywait on ID_modeval\n", __FILE__, __LINE__);
            fflush(stdout);
        }
    }

    free(arraytmp);


	printf("OUTPUT STREAM : %s  ID: %ld\n", IDmodes_val_name, ID_modeval);
	list_image_ID();


    if(MODEVALCOMPUTE == 1) {
        int deviceCount;

        cudaGetDeviceCount(&deviceCount);
        printf("%d devices found\n", deviceCount);
        fflush(stdout);
        printf("\n");
        for(k = 0; k < deviceCount; k++) {
            cudaGetDeviceProperties(&deviceProp, k);
            printf("Device %d / %d [ %20s ]  has compute capability %d.%d.\n",
                   k, deviceCount, deviceProp.name, deviceProp.major, deviceProp.minor);
            printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
            printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
            printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
            printf("\n");
        }


        if(GPUindex < deviceCount) {
            cudaSetDevice(GPUindex);
        } else {
            printf("Invalid Device : %d / %d\n", GPUindex, deviceCount);
            exit(0);
        }


        printf("Create cublas handle ...");
        fflush(stdout);
        cublas_status = cublasCreate(&cublasH);
        if(cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            return EXIT_FAILURE;
        }
        printf(" done\n");
        fflush(stdout);


        // load modes to GPU
        cudaStat = cudaMalloc((void **)&d_modes, sizeof(float) * m * NBmodes);
        if(cudaStat != cudaSuccess) {
            printf("cudaMalloc d_modes returned error code %d, line %d\n", cudaStat, __LINE__);
            exit(EXIT_FAILURE);
        }
        cudaStat = cudaMemcpy(d_modes, data.image[IDmodes].array.F, sizeof(float) * m * NBmodes, cudaMemcpyHostToDevice);
        if(cudaStat != cudaSuccess) {
            printf("cudaMemcpy returned error code %d, line %d\n", cudaStat, __LINE__);
            exit(EXIT_FAILURE);
        }


        // create d_in
        cudaStat = cudaMalloc((void **)&d_in, sizeof(float) * m);
        if(cudaStat != cudaSuccess) {
            printf("cudaMalloc d_in returned error code %d, line %d\n", cudaStat, __LINE__);
            exit(EXIT_FAILURE);
        }


        // create d_modeval
        cudaStat = cudaMalloc((void **)&d_modeval, sizeof(float) * NBmodes);
        if(cudaStat != cudaSuccess) {
            printf("cudaMalloc d_modeval returned error code %d, line %d\n", cudaStat, __LINE__);
            exit(EXIT_FAILURE);
        }
    }



    loopcnt = 0;





    if(TRACEMODE == 1) {
        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        sprintf(traceim_name, "%s_trace", IDmodes_val_name);
        sizearraytmp[0] = TRACEsize;
        sizearraytmp[1] = NBmodes;
        IDtrace = image_ID(traceim_name);
        imOK = 1;
        if(IDtrace == -1) {
            imOK = 0;
        } else {
            if((data.image[IDtrace].md[0].size[0] != TRACEsize) || (data.image[IDtrace].md[0].size[1] != NBmodes)) {
                imOK = 0;
                delete_image_ID(traceim_name);
            }
        }
        if(imOK == 0) {
            IDtrace = create_image_ID(traceim_name, 2, sizearraytmp, _DATATYPE_FLOAT, 1, 0);
        }
        COREMOD_MEMORY_image_set_createsem(traceim_name, 10);
        free(sizearraytmp);
    }





    if(PROCESS == 1) {
        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        sprintf(process_ave_name, "%s_ave", IDmodes_val_name);
        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocave = image_ID(process_ave_name);
        imOK = 1;
        if(IDprocave == -1) {
            imOK = 0;
        } else {
            if((data.image[IDprocave].md[0].size[0] != NBmodes) || (data.image[IDprocave].md[0].size[1] != NBaveSTEP)) {
                imOK = 0;
                delete_image_ID(process_ave_name);
            }
        }
        if(imOK == 0) {
            IDprocave = create_image_ID(process_ave_name, 2, sizearraytmp, _DATATYPE_FLOAT, 1, 0);
        }
        COREMOD_MEMORY_image_set_createsem(process_ave_name, 10);
        free(sizearraytmp);

        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        sprintf(process_rms_name, "%s_rms", IDmodes_val_name);
        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocrms = image_ID(process_rms_name);
        imOK = 1;
        if(IDprocrms == -1) {
            imOK = 0;
        } else {
            if((data.image[IDprocrms].md[0].size[0] != NBmodes) || (data.image[IDprocrms].md[0].size[1] != NBaveSTEP)) {
                imOK = 0;
                delete_image_ID(process_rms_name);
            }
        }
        if(imOK == 0) {
            IDprocrms = create_image_ID(process_rms_name, 2, sizearraytmp, _DATATYPE_FLOAT, 1, 0);
        }
        COREMOD_MEMORY_image_set_createsem(process_rms_name, 10);
        free(sizearraytmp);
    }


    initref = 0;






    twait1 = *twait;

    printf("LOOP START   MODEVALCOMPUTE = %d\n", MODEVALCOMPUTE);
    fflush(stdout);

    if(MODEVALCOMPUTE == 0) {
        printf("\n");
        printf("This function is NOT computing mode values\n");
        printf("Pre-existing stream %s was detected\n", IDmodes_val_name);
        printf("\n");
        if(data.processinfo == 1) {
            strcpy(processinfo->statusmsg, "Passing stream, no computation");
            //sprintf(processinfo->description, "passthrough, no comp");
        }
    } else {
        char msgstring[200];
        sprintf(msgstring, "Running on GPU %d", GPUindex);
        if(data.processinfo == 1) {
            strcpy(processinfo->statusmsg, msgstring);
        }
    }







    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(processinfo); // Notify processinfo that we are entering loop

    if(MODEVALCOMPUTE == 1) {
        sprintf(pinfomsg, "MVM %s %s -> %s TRACE=%d PROC=%d", IDmodes_name, in_stream, IDmodes_val_name, TRACEMODE, PROCESS);
    } else {
        sprintf(pinfomsg, "passthrough %s TRACE=%d PROC=%d", IDmodes_val_name, TRACEMODE, PROCESS);
    }
    processinfo_WriteMessage(processinfo, pinfomsg);



    while(loopOK == 1) {
        struct timespec tdiff;
        double tdiffv;

        int t00OK = 0;
        int t01OK = 0;
        int t02OK = 0;
        int t03OK = 0;
        int t04OK = 0;
        int t05OK = 0;
        int t06OK = 0;

        loopOK = processinfo_loopstep(processinfo);

        clock_gettime(CLOCK_REALTIME, &t0);


        // We either compute the result in this function (MODEVALCOMPUTE = 1)
        // or we read it from ID_modeval stream (MODEVALCOMPUTE = 0)

        if(MODEVALCOMPUTE == 1) {

            // Are we computing a new reference ?
            // if yes, set initref to 0 (reference is NOT initialized)
            //
            if(refindex != data.image[IDref].md[0].cnt0) {
                initref = 0;
                refindex = data.image[IDref].md[0].cnt0;
            }


            if(initref == 1) {
                // Reference is already initialized
                // wait for input stream to be changed to start computation
                //
                if(data.image[IDin].md[0].sem == 0) {
                    // if not using semaphore, use counter #0
                    while(data.image[IDin].md[0].cnt0 == cnt) { // test if new frame exists
                        usleep(5);
                    }
                    cnt = data.image[IDin].md[0].cnt0;
                    semr = 0;
                } else {
                    // if using semaphore

                    // we wait for 1 sec max
                    if(clock_gettime(CLOCK_REALTIME, &ts) == -1) {
                        perror("clock_gettime");
                        exit(EXIT_FAILURE);
                    }
                    ts.tv_sec += 1;
                    semr = sem_timedwait(data.image[IDin].semptr[insem], &ts);

                    // drive semaphore to zero if it isn't already
                    while(sem_trywait(data.image[IDin].semptr[insem]) == 0) {
                        if(semwarn == 1) {
                            int semval;
                            sem_getvalue(data.image[IDin].semptr[insem], &semval);
                            printf("WARNING %s %d  : sem_trywait on IDin  seval = %d\n", __FILE__, __LINE__, semval);
                            fflush(stdout);
                        }
                    }

                }
            } else { // compute response of reference immediately
                printf("COMPUTE NEW REFERENCE RESPONSE\n");
                semr = 0;
            }

            t00OK = 1;
            clock_gettime(CLOCK_REALTIME, &t00);

            processinfo_exec_start(processinfo);

            if(semr == 0) {
                // load in_stream to GPU
                if(initref == 0) {
                    cudaStat = cudaMemcpy(d_in, data.image[IDref].array.F, sizeof(float) * m, cudaMemcpyHostToDevice);
                } else {
                    cudaStat = cudaMemcpy(d_in, data.image[IDin].array.F, sizeof(float) * m, cudaMemcpyHostToDevice);
                }


                if(cudaStat != cudaSuccess) {
                    printf("initref = %d    %ld  %ld\n", initref, IDref, IDin);
                    printf("cudaMemcpy returned error code %d, line %d\n", cudaStat, __LINE__);
                    exit(EXIT_FAILURE);
                }

                t01OK = 1;
                clock_gettime(CLOCK_REALTIME, &t01);

                if(BETAMODE == 1) {
                    beta = -1.0;
                    cudaStat = cudaMemcpy(d_modeval, modevalarrayref, sizeof(float) * NBmodes, cudaMemcpyHostToDevice);
                }

                t02OK = 1;
                clock_gettime(CLOCK_REALTIME, &t02);

                // compute
                cublas_status = cublasSgemv(cublasH, CUBLAS_OP_T, m, NBmodes, &alpha, d_modes, m, d_in, 1, &beta, d_modeval, 1);
                if(cublas_status != CUBLAS_STATUS_SUCCESS) {
                    printf("cublasSgemv returned error code %d, line(%d)\n", cublas_status, __LINE__);
                    fflush(stdout);
                    if(cublas_status == CUBLAS_STATUS_NOT_INITIALIZED) {
                        printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_INVALID_VALUE) {
                        printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_ARCH_MISMATCH) {
                        printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_EXECUTION_FAILED) {
                        printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                    }

                    printf("GPU index                           = %d\n", GPUindex);

                    printf("CUBLAS_OP                           = %d\n", CUBLAS_OP_T);
                    printf("alpha                               = %f\n", alpha);
                    printf("alpha                               = %f\n", beta);
                    printf("m                                   = %d\n", (int) m);
                    printf("NBmodes                             = %d\n", (int) NBmodes);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }

                // copy result
                data.image[ID_modeval].md[0].write = 1;

                t03OK = 1;
                clock_gettime(CLOCK_REALTIME, &t03);

                if(initref == 0) { // construct reference to be subtracted
					printf("... reference compute\n");
                    cudaStat = cudaMemcpy(modevalarrayref, d_modeval, sizeof(float) * NBmodes, cudaMemcpyDeviceToHost);

                    IDrefout = image_ID(IDrefout_name);
                    if(IDrefout != -1)
                        for(k = 0; k < NBmodes; k++) {
                            modevalarrayref[k] -= data.image[IDrefout].array.F[k];
                        }


                    if((INNORMMODE == 0) && (MODENORM == 0)) {
                        BETAMODE = 1;    // include ref subtraction in GPU operation
                    } else {
                        BETAMODE = 0;
                    }
                } else {
                    cudaStat = cudaMemcpy(modevalarray, d_modeval, sizeof(float) * NBmodes, cudaMemcpyDeviceToHost);

                    if(BETAMODE == 0) {
                        for(k = 0; k < NBmodes; k++) {
                            data.image[ID_modeval].array.F[k] = (modevalarray[k] / data.image[IDintot].array.F[0] - modevalarrayref[k]) / normcoeff[k];
                        }
                    } else
                        for(k = 0; k < NBmodes; k++) {
                            data.image[ID_modeval].array.F[k] = modevalarray[k];
                        }


                    COREMOD_MEMORY_image_set_sempost_byID(ID_modeval, -1);

                    data.image[ID_modeval].md[0].cnt0++;
                    data.image[ID_modeval].md[0].write = 0;
                }
            }                        
        } else { // WAIT FOR NEW MODEVAL
            int rval;
            rval = sem_wait(data.image[ID_modeval].semptr[insem]);
            if(rval == -1) // interrupt
				loopOK = 0;
            
            processinfo_exec_start(processinfo);
        }




        t04OK = 1;
        clock_gettime(CLOCK_REALTIME, &t04);



        if(TRACEMODE == 1) {
            data.image[ID_modeval].md[0].write = 1;

            for(k = 0; k < NBmodes; k++) {
                data.image[IDtrace].array.F[k * TRACEsize + TRACEindex] = data.image[ID_modeval].array.F[k];
            }
            data.image[IDtrace].md[0].cnt1 = TRACEindex;

            sem_getvalue(data.image[IDtrace].semptr[0], &semval);
            if(semval < SEMAPHORE_MAXVAL) {
                sem_post(data.image[IDtrace].semptr[0]);
            }
            sem_getvalue(data.image[IDtrace].semptr[1], &semval);
            if(semval < SEMAPHORE_MAXVAL) {
                sem_post(data.image[IDtrace].semptr[1]);
            }
            data.image[IDtrace].md[0].cnt0++;
            data.image[IDtrace].md[0].write = 0;

            TRACEindex++;
            if(TRACEindex >= TRACEsize) {
                TRACEindex = 0;
                // copy to tracef shared memory (frozen trace)
            }
        }

        t05OK = 1;
        clock_gettime(CLOCK_REALTIME, &t05);

        if(PROCESS == 1) {
            stepcoeff = stepcoeff0;
            data.image[IDprocave].md[0].write = 1;
            for(step = 0; step < NBaveSTEP; step++) {
                for(k = 0; k < NBmodes; k++) {
                    data.image[IDprocave].array.F[NBmodes * step + k] = (1.0 - stepcoeff) * data.image[IDprocave].array.F[NBmodes * step + k] + stepcoeff * data.image[ID_modeval].array.F[k];
                }
                stepcoeff *= stepcoeff0;
            }
            for(semnb = 0; semnb < data.image[IDprocave].md[0].sem; semnb++) {
                sem_getvalue(data.image[IDprocave].semptr[semnb], &semval);
                if(semval < SEMAPHORE_MAXVAL) {
                    sem_post(data.image[IDprocave].semptr[semnb]);
                }
            }
            data.image[IDprocave].md[0].cnt0++;
            data.image[IDprocave].md[0].write = 0;

            stepcoeff = stepcoeff0;
            data.image[IDprocrms].md[0].write = 1;
            for(step = 0; step < NBaveSTEP; step++) {
                for(k = 0; k < NBmodes; k++) {
                    tmpv = data.image[ID_modeval].array.F[k] - data.image[IDprocave].array.F[NBmodes * step + k];
                    tmpv = tmpv * tmpv;
                    data.image[IDprocrms].array.F[NBmodes * step + k] = (1.0 - stepcoeff) * data.image[IDprocrms].array.F[NBmodes * step + k] + stepcoeff * tmpv;
                }
                stepcoeff *= stepcoeff0;
            }
            for(semnb = 0; semnb < data.image[IDprocrms].md[0].sem; semnb++) {
                sem_getvalue(data.image[IDprocrms].semptr[semnb], &semval);
                if(semval < SEMAPHORE_MAXVAL) {
                    sem_post(data.image[IDprocrms].semptr[semnb]);
                }
            }
            data.image[IDprocrms].md[0].cnt0++;
            data.image[IDprocrms].md[0].write = 0;
        }

        t06OK = 1;
        clock_gettime(CLOCK_REALTIME, &t06);


        processinfo_exec_end(processinfo);


        if(twait1 < 0) {
            twait1 = 0;
        }

        if(*twait > 0) {
            usleep(twait1);
        }

        clock_gettime(CLOCK_REALTIME, &t1);
        tdiff = info_time_diff(t0, t1);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv < 1.0e-6 * (*twait)) {
            twait1 ++;
        } else {
            twait1 --;
        }
        //TEST TIMING
        /*
        		if(tdiffv>1.0e-3)
        		{
        			printf("  ... function CUDACOMP_extractModesLoop - TIMING GLITCH  [%09ld] [%d]\n", t0.tv_nsec, insem);
        			printf("       %ld   timing info : %11.9lf  %ld %ld\n", iter, tdiffv, twait1, twait);
        			fflush(stdout);

        			tdiff = info_time_diff(t0, t00);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t00: %8.3lf\n", t00OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t01);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t01: %8.3lf\n", t01OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t02);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t02: %8.3lf\n", t02OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t03);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t03: %8.3lf\n", t03OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t04);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t04: %8.3lf\n", t04OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t05);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t05: %8.3lf\n", t05OK, 1.0e3*tdiffv);

        			tdiff = info_time_diff(t0, t06);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf(" %d            t06: %8.3lf\n", t06OK, 1.0e3*tdiffv);

        			printf("---\n");



        			tdiff = info_time_diff(t0, t04);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf("             0  -> 04: %8.3lf\n", 1.0e3*tdiffv);

        			tdiff = info_time_diff(t04, t05);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf("             04 -> 05: %8.3lf\n", 1.0e3*tdiffv);

        			tdiff = info_time_diff(t05, t06);
        			tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        			printf("             05 -> 06: %8.3lf\n", 1.0e3*tdiffv);

        		}
        	*/



        initref = 1;
    }

    processinfo_cleanExit(processinfo);



    if(MODEVALCOMPUTE == 1) {
        cudaFree(d_modes);
        cudaFree(d_in);
        cudaFree(d_modeval);



        if(cublasH) {
            cublasDestroy(cublasH);
        }
    }

    free(normcoeff);
    free(modevalarray);
    free(modevalarrayref);





    return RETURN_SUCCESS;
}











/** @brief extract mode coefficients from data stream (MVM)
 *
 */

int  __attribute__((hot)) CUDACOMP_MVMextractModesLoop(
    const char *in_stream,           // input stream
    const char *intot_stream,        // [optional]   input normalization stream
    const char *IDmodes_name,        // Modes matrix
    const char *IDrefin_name,        // [optional] input reference  - to be subtracted
    const char *IDrefout_name,       // [optional] output reference - to be added
    const char *IDmodes_val_name,    // ouput stream
    int         GPUindex,            // GPU index
    int         PROCESS,             // 1 if postprocessing
    int         TRACEMODE,           // 1 if writing trace
    int         MODENORM,            // 1 if input modes should be normalized
    int         insem,               // input semaphore index
    int         axmode,              // 0 for normal mode extraction, 1 for expansion
    long        twait,               // if >0, insert time wait [us] at each iteration
    int         semwarn              // 1 if warning when input stream semaphore >1
) {


    // ==================================
    // CREATE FPS AND START CONF
    // ==================================

    char fpsname[200];
    long pindex = (long) getpid();  // index used to differentiate multiple calls to function
    // if we don't have anything more informative, we use PID
    FUNCTION_PARAMETER_STRUCT fps;
    sprintf(fpsname, "cudaMVMextmodes-%06ld", pindex);
    CUDACOMP_MVMextractModesLoop_FPCONF(fpsname, CMDCODE_FPSINIT);




    // ==================================
    // SET PARAMETER VALUES
    // ==================================

    function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_SIMPLE);

    functionparameter_SetParamValue_STRING(&fps, ".sname_in",            in_stream);
    functionparameter_SetParamValue_STRING(&fps, ".sname_modes",         IDmodes_name);
    functionparameter_SetParamValue_STRING(&fps, ".option.sname_intot",  intot_stream);
    functionparameter_SetParamValue_STRING(&fps, ".option.sname_refin",  IDrefin_name);
    functionparameter_SetParamValue_STRING(&fps, ".option.sname_refout", IDrefout_name);
    functionparameter_SetParamValue_STRING(&fps, ".sname_outmodesval",   IDmodes_val_name);

    functionparameter_SetParamValue_INT64(&fps, ".GPUindex",             GPUindex);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.PROCESS",       PROCESS);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.TRACEMODE",     TRACEMODE);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.MODENORM",      MODENORM);
    functionparameter_SetParamValue_INT64(&fps, ".option.insem",         insem);
    functionparameter_SetParamValue_INT64(&fps, ".option.axmode",        axmode);
    functionparameter_SetParamValue_INT64(&fps, ".option.twait",         twait);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.semwarn",       semwarn);

    function_parameter_struct_disconnect(&fps);




    // ==================================
    // START RUN PROCESS
    // ==================================

    CUDACOMP_MVMextractModesLoop_RUN(fpsname);


    return RETURN_SUCCESS;
}









	




#endif










