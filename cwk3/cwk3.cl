__kernel void heat(__global float *grid, __global float *new_grid, int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    // since we run the problem so it fits nicely in big groups, we will have
    // redundant instances so skip them so we don't go out of bounds
    if (i >= N || j >= N)
        return;

    // apply the heat rule on the problem size
    if (i == 0 || j == 0 || i == N - 1 || j == N - 1) {
        new_grid[i * N + j] = 0;
    } else {
        new_grid[i * N + j] = 0.25 * (
            grid[i * N + (j - 1)] + grid[i * N + (j + 1)] +
            grid[(i + 1) * N + j] + grid [(i - 1) * N + j]
        );
    }
}
