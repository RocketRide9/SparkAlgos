#define real double
#define real4 float4

kernel void SymDiagMul(
    global const real *d3,
    global const real *d2,
    global const real *d1,
    global const real *d0,
    
    global const real *di,    

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
        if (t >= 0) dot += d3[t] * v[t];
        t = row - 2 - gap;
        if (t >= 0) dot += d2[t] * v[t];
        t = row - 1 - gap;
        if (t >= 0) dot += d1[t] * v[t];
        t = row - 1;
        if (t >= 0) dot += d0[t] * v[t];
        
        dot += di[row] * v[row];

        t = row+1;
        if (t < n) dot += d0[row] * v[t];
        t = row+1+gap;
        if (t < n) dot += d1[row] * v[t];
        t = row+2+gap;
        if (t < n) dot += d2[row] * v[t];
        t = row+3+gap;
        if (t < n) dot += d3[row] * v[t];
        
        res[row] = dot;
    }
}
