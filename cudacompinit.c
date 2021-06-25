/** @file cudacompinit.c
 */


#ifdef HAVE_MAGMA
#include "magma_v2.h"
#include "magma_lapack.h"
#endif



#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


#ifdef HAVE_CUDA


extern int cuda_deviceCount;



// ==========================================
// Forward declaration(s)
// ==========================================

int CUDACOMP_init();


// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t delete_image_ID__cli()
{
    long i = 1;
    printf("%ld : %d\n", i, data.cmdargtoken[i].type);
    while(data.cmdargtoken[i].type != 0)
    {
        if(data.cmdargtoken[i].type == 4)
        {
            delete_image_ID(data.cmdargtoken[i].val.string, DELETE_IMAGE_ERRMODE_WARNING);
        }
        else
        {
            printf("Image %s does not exist\n", data.cmdargtoken[i].val.string);
        }
        i++;
    }

    return CLICMD_SUCCESS;
}




// ==========================================
// Register CLI command(s)
// ==========================================

errno_t cudacompinit_addCLIcmd()
{

    RegisterCLIcommand(
        "cudacompinit",
        __FILE__,
        CUDACOMP_init,
        "init CUDA comp",
        "no argument",
        "cudacompinit",
        "int CUDACOMP_init()"
    );


    return RETURN_SUCCESS;
}





/**
 * @brief Initialize CUDA and MAGMA
 *
 * Finds CUDA devices
 * Initializes CUDA and MAGMA libraries
 *
 * @return number of CUDA devices found
 *
 */
int CUDACOMP_init()
{
    int device;
    struct cudaDeviceProp deviceProp;

    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%d devices found\n", cuda_deviceCount);
    printf("\n");
    for(device = 0; device < cuda_deviceCount; ++device)
    {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d [ %20s ]  has compute capability %d.%d.\n",
               device, deviceProp.name, deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("\n");
#ifdef HAVE_MAGMA
        printf("Using MAGMA library\n");
        magma_print_environment();
#endif

        printf("\n");
    }

    return((int) cuda_deviceCount);
}









void *GPU_scanDevices(void *deviceCount_void_ptr)
{
    int *devcnt_ptr = (int *) deviceCount_void_ptr;
    int device;
    struct cudaDeviceProp deviceProp;

    printf("Scanning for GPU devices ...\n");
    fflush(stdout);

    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%d devices found\n", cuda_deviceCount);
    fflush(stdout);

    printf("\n");
    for(device = 0; device < cuda_deviceCount; ++device)
    {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d [ %20s ]  has compute capability %d.%d.\n",
               device, deviceProp.name, deviceProp.major, deviceProp.minor);

        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long) deviceProp.totalGlobalMem);

        printf("  (%2d) Multiprocessors\n",
               deviceProp.multiProcessorCount);

        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
               deviceProp.clockRate * 1e-3f,
               deviceProp.clockRate * 1e-6f);

        printf("\n");
    }

    printf("Done scanning for GPU devices\n");
    fflush(stdout);

    *devcnt_ptr = cuda_deviceCount;

    return NULL;
}










#endif
