__kernel void heat(__global float *grid, __global float *new_grid, int N)
{
    // + 1 because i set the group size with (N - 2) in each direction
    // so that we skip the edges
    size_t i = get_global_id(0) + 1;
    size_t j = get_global_id(1) + 1;

    new_grid[i * N + j] = 0.25 * (
        grid[i * N + (j - 1)] + grid[i * N + (j + 1)] +
        grid[(i + 1) * N + j] + grid [(i - 1) * N + j]
    );
}
