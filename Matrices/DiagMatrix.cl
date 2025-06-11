#define real double
#define real4 float4

kernel void DiagMul(
    global const real *ld3,
    global const real *ld2,
    global const real *ld1,
    global const real *ld0,
    
    global const real *di,
    
    global const real *rd0,
    global const real *rd1,
    global const real *rd2,
    global const real *rd3,

    const int n,
    const int gap,
    
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        real dot = 0;
        
        int t = row - 3 - gap;
        if (t >= 0) dot += ld3[t] * v[t];
        t = row - 2 - gap;
        if (t >= 0) dot += ld2[t] * v[t];
        t = row - 1 - gap;
        if (t >= 0) dot += ld1[t] * v[t];
        t = row - 1;
        if (t >= 0) dot += ld0[t] * v[t];
        
        dot += di[row] * v[row];

        t = row+1;
        if (t < n) dot += rd0[row] * v[t];
        t = row+1+gap;
        if (t < n) dot += rd1[row] * v[t];
        t = row+2+gap;
        if (t < n) dot += rd2[row] * v[t];
        t = row+3+gap;
        if (t < n) dot += rd3[row] * v[t];
        
        res[row] = dot;
    }
}
