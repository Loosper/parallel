Complete the table below with your results, and then provide your interpretation at the end.

Note that:

- When calculating the parallel speed-up S, use the time output by the code, which corresponds
  to the parallel calculation and does not include initialising the matrix/vectors or
  performing the serial check.

- Take as the serial execution time the time output by the code when run with a single process
  (hence the speed-up for 1 process should be 1.0).


No. Process:                        Mean time (average of 3 runs)           Parallel speed-up, S:
===========                         ============================:           ====================
1                                   3.70601                                 1.0
2                                   2.36303                                 1.568
3                                   2.12142                                 1.749
4                                   1.98982                                 1.862

Architecture that the timing runs were performed on:
    AMD Ryzen 7 PRO 4750U with N=30000

    It is clear from that for a fixed problem size the speedup increase is
    logarithmic and therefore increasing the number of processes gives
    diminishing returns. This is consistent with Amdahl's law, as a large
    fraction of work to be done is communicating the problem itself to all
    other processes.
