#include "CommandLineInterface/CLIcore.h"


// Local variables pointers
static uint32_t *GPUindex;
long fpi_GPUindex;




static CLICMDARGDEF farg[] =
{
    {
        CLIARG_UINT32, ".GPUindex", "GPU index", "0",
        CLIARG_VISIBLE_DEFAULT, (void **) &GPUindex, &fpi_GPUindex
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

