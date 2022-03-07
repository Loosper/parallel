//
// COMP/XJCO3221 Parallel Computation Coursework 2: MPI / Distributed Memory
// Parallelism.
//


//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// The MPI library.
#include <mpi.h>

// The include file for this coursework. Do not change the routines in
// "cwk2_extra.h", or avoid calling them, as theyt will be changed with
// different versions for the purposes of assessment.
#include "cwk2_extra.h"

void do_calculation(int rows, int cols, float *A, float *x, float *b)
{
	int row, col;
	for (row = 0; row < rows; row++) {
		b[row] = 0.0f;
		for (col = 0; col < cols; col++)
			b[row] += A[row * cols + col] * x[col];
	}
}

int main(int argc, char *argv[])
{
	//
	// Initialisation.
	//

	// Start MPI and get the rank of this process.
	int numprocs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// The matrix is of size NxN, and the vectors are of size N, where N is
	// specified on the command line, so need to first get N. The function
	// 'getNFromCommandLine' does this, and also checks that N is divisible
	// by the number of processes. If it returns -1, something went wrong
	// and MPI_Finalize() has already been called on all ranks.

	// Do NOT modify this call or the function in cwk2_extra.h
	int N = getNFromCommandLine(argc, argv, rank, numprocs);
	if (N == -1)
		return EXIT_FAILURE;

	// The number of matrix rows per process. Note we have checked by this
	// point that N is divisible by numprocs.
	int rowsPerProc = N / numprocs;
	// All processes use the full vector x, i.e. the one that is being
	// multiplied. Allocate memory for it - no error check for simplicity.
	float *x = (float *) malloc(N * sizeof(float));
	// All processes also have an equal number of matrix rows, and a
	// portion of the solution vector, so allocate memory for these as
	// well.
	size_t A_size = N * rowsPerProc;
	float *A_perProc = (float *) malloc(A_size      * sizeof(float));
	float *b_perProc = (float *) malloc(rowsPerProc * sizeof(float));

	// Allocate the full matrix and the full solution vector on rank 0
	// only. Also initialise A and x with values.
	float *A = NULL;
	float *b = NULL;
	if (rank == 0)
		// Do NOT modify this call or the function in cwk2_extra.h
		initialiseMatrixAndSolution(&A, &b, x, N);

	// Start the timing.
	double startTime = MPI_Wtime();

	//
	// Perform matrix-vector multiplication in parallel.
	//

	if (numprocs && ((numprocs & (numprocs - 1)) == 0)) {
		// TODO: binary tree
		MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	} else {
		// just scatter it otherwise
		MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	MPI_Scatter(
		A,         A_size, MPI_FLOAT,
		A_perProc, A_size, MPI_FLOAT,
		0, MPI_COMM_WORLD
	);
	MPI_Scatter(
		b,         rowsPerProc, MPI_FLOAT,
		b_perProc, rowsPerProc, MPI_FLOAT,
		0, MPI_COMM_WORLD
	);

	do_calculation(rowsPerProc, N, A_perProc, x, b_perProc);

	MPI_Gather(
		b_perProc, rowsPerProc, MPI_FLOAT,
		b,         rowsPerProc, MPI_FLOAT,
		0, MPI_COMM_WORLD
	);

	//
	// Check the answer on rank 0 in serial. Also output the result of the
	// timing.
	//
	if (rank == 0) {
		// Time for the parallel computation.
		printf(
			"Parallel calculation took a total time %g s.\n",
			MPI_Wtime() - startTime
		);
		// Call the check routine "checkAgainstSerial()" in
		// cwk2_extra.h, which prints messages explaining if the
		// parallel calculation matches the serial one, to within
		// floating-point precision.
		// Do NOT modify this call or the function in cwk2_extra.h
		checkAgainstSerial(A, x, b, N);
	}

	//
	// Finalise.
	//
	// Free up all memory allocated for x, A_perProc and b_perProc, and (on
	// rank 0 only) A and b.
	freeAllMemory(rank, A, b, x, A_perProc, b_perProc);
	// Tell MPI to free up resources.
	MPI_Finalize();
}
