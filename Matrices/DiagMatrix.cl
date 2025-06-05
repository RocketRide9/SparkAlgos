#define real double
#define real4 float4

kernel void DIAGMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    const int n,
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        int start = aptr[row];
        int stop = aptr[row + 1];
        real dot = di[row]*v[row];
        for (int a = start; a < stop; a++)
        {
            dot += mat[a]*v[jptr[a]];
        }
        res[row] = dot;
    }
}
