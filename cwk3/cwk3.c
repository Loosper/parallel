//
// Starting point for the OpenCL coursework for COMP/XJCO3221 Parallel Computation.
//
// Once compiled, execute with the size of the square grid as a command line argument, i.e.
//
// ./cwk3 16
//
// will generate a 16 by 16 grid. The C-code below will then display the initial grid,
// followed by the same grid again. You will need to implement OpenCL that applies the heat
// equation as per the instructions, so that the final grid is different.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 2 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// getCmdLineArg()  :  Parses grid size N from command line argument, or fails with error message.
// fillGrid()       :  Fills the grid with random values, except boundary values which are always zero.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{

    //
    // Parse command line argument and check it is valid. Handled by a routine in the helper file.
    //
    int N;
    getCmdLineArg( argc, argv, &N );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    size_t gridSize = N * N * sizeof(float);
    // Allocate memory for the grid. For simplicity, this uses a one-dimensional array.
    float *hostGrid = (float*) malloc(gridSize);

    // Fill the grid with some initial values, and display to stdout. fillGrid() is defined in the helper file.
    fillGrid( hostGrid, N );
    printf( "Original grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

    cl_mem deviceInGrid = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gridSize, hostGrid,
        &status
    );

    cl_mem deviceOutGrid = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY, gridSize, NULL,
        &status
    );

    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "heat", context, device);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceInGrid);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceOutGrid);

    size_t globalSize[2]    = {};
    size_t workGroupSize[2] = {};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, workGroupSize, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        printf("Failed to enqueue kernel: %d.\n", status);
        return EXIT_FAILURE;
    }

    status = clEnqueueReadBuffer(queue, deviceOutGrid, CL_TRUE, 0, gridSize, hostGrid, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        printf("Failed copying data back: %d.\n", status);
        return EXIT_FAILURE;
    }

    clReleaseMemObject(deviceInGrid);
    clReleaseMemObject(deviceOutGrid);
    clReleaseKernel(kernel);

    //
    // Display the final result. This assumes that the iterated grid was copied back to the hostGrid array.
    //
    printf( "Final grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );

    free( hostGrid );

    return EXIT_SUCCESS;
}
