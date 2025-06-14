#define real double
#define real4 float4

kernel void dot_kernel9999
(
    global const real4 *v1,
    global const real4 *v2,
    const int n,
    global real *res,
    local real *work
)
{
    // Compute partial dot product
    real sum = 0;
    for (int k=get_global_id(0);k<n;k+=get_global_size(0))
    {
        // sum += a[get_global_id(ROW_DIM)+m*k] * x[k];
        sum += dot(v1[k], v2[k]);
    }

    // Each thread stores its partial sum in WORK
    int cols = get_local_size(0); // initial cols in group
    // int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
    int jj = get_local_id(0); // block index in column, 0<=jj<cols
    work[jj] = sum;
    barrier(CLK_LOCAL_MEM_FENCE); // sync group

    // Reduce sums in log2(cols) steps
    while ( cols > 1 )
    {
        cols /= 2;
        if (jj < cols) work[jj] += work[jj+cols];
        barrier(CLK_LOCAL_MEM_FENCE); // sync group
    }

    // Write final result in Y
    if ( jj == 0 ) *res = work[0];
}

// y = y^-1/2
kernel void BLAS_rsqrt
(
    global real *y,
    const int n
)
{
    uint i = get_global_id(0);
    if (i < n)
    {
        y[i] = rsqrt(y[i]);
    }
}

kernel void VecMul
(
    global real *y,
    const global real *x,
    const int n
)
{
    #pragma unroll
    for (int w = 0; w < 4; w++)
    {
        uint i = w*get_global_size(0) + get_global_id(0);
        if (i < n)
        {
            y[i] *= x[i];
        }
    }
}

kernel void BiCGSTAB_p
(
    global real *p,
    global const real *r,
    global const real *nu,
    const real w,
    const real beta,
    const int n
)
{
    uint i = get_global_id(0);
    if (i < n)
    {
        p[i] = r[i] + beta * (p[i] - w*nu[i]);
    }
}
