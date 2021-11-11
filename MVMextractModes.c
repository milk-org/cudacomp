#include "CommandLineInterface/CLIcore.h"


// Local variables pointers
static uint32_t *GPUindex;
long fpi_GPUindex;

static char *insname;
long fpi_insname;

static char *immodes;
long fpi_immodes;

static char *outcoeff;
long fpi_outcoeff;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_UINT32, ".GPUindex", "GPU index", "0",
        CLIARG_VISIBLE_DEFAULT, (void **) &GPUindex, &fpi_GPUindex
    },
    {
        CLIARG_STREAM, ".insname", "input stream name", "null",
        CLIARG_VISIBLE_DEFAULT, (void **) &insname, &fpi_insname
    },
    {
        CLIARG_STREAM, ".immodes", "modes stream name", "null",
        CLIARG_VISIBLE_DEFAULT, (void **) &immodes, &fpi_immodes
    },
    {
        CLIARG_STREAM, ".outcoeff", "output coefficients", "null",
        CLIARG_VISIBLE_DEFAULT, (void **) &outcoeff, &fpi_outcoeff
    }
};



// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}


// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "MVMmextrmodes",
    "extract modes from WFS",
    CLICMD_FIELDS_DEFAULTS
};


// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to input streams
    IMGID imgin = makeIMGID(insname);
    resolveIMGID(&imgin, ERRMODE_ABORT);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);


    INSERT_STD_PROCINFO_COMPUTEFUNC_START



    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_cudacomp__MVMextractModes()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

