#define real float

// original: https://github.com/CNugteren/CLBlast/blob/bd96941ac0633e8e7d09fd2475e0279be370b1e1/src/kernels/level1/xdot.opencl

#define WGS1 32
#define WGS2 32

kernel
void Xdot(const int n,
          const global real* restrict xgm,
          const global real* restrict ygm,
          global real* output)
{
    local real lm[WGS1];
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int num_groups = get_num_groups(0);
    // Performs multiplication and the first steps of the reduction
    real acc = 0;
    int id = wgid*WGS1 + lid;
    while (id < n) {
        real x = xgm[id];
        real y = ygm[id];
        acc += x * y;
        id += WGS1*num_groups;
    }
    lm[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Performs reduction in local memory
    for (int s=WGS1/2; s>0; s = s >> 1) {
        if (lid < s) {
            lm[lid] = lm[lid] + lm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Stores the per-workgroup result
    if (lid == 0) {
        output[wgid] = lm[0];
    }
}


// The epilogue reduction kernel, performing the final bit of the sum operation. This kernel has to
// be launched with a single workgroup only.
kernel
void XdotEpilogue(const global real* restrict input,
                  global real* dot) {
    local real lm[WGS2];
    const int lid = get_local_id(0);
    
    // Performs the first step of the reduction while loading the data
    lm[lid] = input[lid] + input[lid + WGS2];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Performs reduction in local memory
    for (int s=WGS2/2; s>0; s=s>>1) {
        if (lid < s) {
            lm[lid] += lm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Stores the final result
    if (lid == 0) {
        dot[0] = lm[0];
    }
}

// y *= a
kernel void BLAS_scale
(
    const real a,
    global real *y,
    const int n
)
{
    uint i = get_global_id(0);
    if (i < n)
    {
        y[i] *= a;
    }
}

// y += a*x
kernel void BLAS_axpy
(
    const real a,
    global const real *x,
    global real *y,
    const int n
)
{
    uint i = get_global_id(0);
    if (i < n)
    {
        y[i] += a * x[i];
    }
}
