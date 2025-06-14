using Real = double;

using System.Diagnostics;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.SlaeSolver;

public class CgmOCL : IDisposable, ISlaeSolver
{
    int _maxIter;
    Real _eps;

    int _n = 0; // размерность СЛАУ
    ComputeBuffer<Real> r;
    ComputeBuffer<Real> di_inv;
    ComputeBuffer<Real> mr;
    ComputeBuffer<Real> az;
    ComputeBuffer<Real> z;
    ComputeBuffer<Real> dotpart;
    ComputeBuffer<Real> dotres;
    private bool disposedValue;

    public CgmOCL(
        int maxIter,
        Real eps
    ) {
        _maxIter = maxIter;
        // TODO: уменьшение eps чтобы невязка ответа была сравнима с BicgStab
        _eps = eps / 1e+7;

        dotpart = new ComputeBuffer<Real>(32*2, BufferFlags.OnDevice);
        dotres  = new ComputeBuffer<Real>(1, BufferFlags.OnDevice);
    }

    public static ISlaeSolver Construct(int maxIter, double eps)
        => new CgmOCL(maxIter, eps);

    // Выделить память для временных массивов
    // n - длина каждого массива
    public void AllocateTemps(int n)
    {
        if (n != _n)
        {
            _n = n;

            r       = new (n, BufferFlags.OnDevice);
            di_inv  = new (n, BufferFlags.OnDevice);
            mr      = new (n, BufferFlags.OnDevice);
            az      = new (n, BufferFlags.OnDevice);
            z       = new (n, BufferFlags.OnDevice);
        }
    }

    static ComputeProgram? solvers;
    public (Real discrep, int iter) Solve(Types.Matrix matrix, Span<Real> b, Span<Real> x)
    {
        var sw = Stopwatch.StartNew();
        AllocateTemps(x.Length);

        var _x = new ComputeBuffer<Real>(x, BufferFlags.OnDevice);
        var _b = new ComputeBuffer<Real>(b, BufferFlags.OnDevice);
        
        var globalWork = new NDRange((nuint)x.Length).PadTo(Core.Prefered1D);
        var localWork = new NDRange(Core.Prefered1D);

        // BiCGSTAB
        if (solvers == null)
        {
            solvers = new ComputeProgram("SlaeSolver/Solvers.cl");
        }

        var kernRsqrt = solvers.GetKernel(
            "BLAS_rsqrt",
            globalWork,
            localWork
        );
        Event RsqrtExecute(ComputeBuffer<Real> _y) {
            kernRsqrt.SetArg(0, _y);
            kernRsqrt.SetArg(1, _y.Length);
            return kernRsqrt.Execute();
        }

        var kernVecMul = solvers.GetKernel(
            "VecMul",
            new NDRange((nuint)_x.Length/4).PadTo(Core.Prefered1D),
            new(Core.Prefered1D)
        );
        Event VecMulExecute(ComputeBuffer<Real> _y, ComputeBuffer<Real> _x) {
            kernVecMul.SetArg(0, _y);
            kernVecMul.SetArg(1, _x);
            kernVecMul.SetArg(2, _y.Length);
            return kernVecMul.Execute();
        }

        // TODO: set scratch buffers once per SBlas instance
        var SBlas = SparkAlgos.Blas.GetInstance();
        SBlas.Scratch64 = dotpart;
        SBlas.Scratch1 = dotres;

        Trace.WriteLine($"OpenCL prepare: {sw.ElapsedMilliseconds}ms");

        // precond
        matrix.Di.CopyDeviceTo(di_inv);
        RsqrtExecute(di_inv);
        // Cgm
        // 1.
        matrix.Mul(_x, z);
        _b.CopyDeviceTo(r);
        SBlas.Axpy(-1, z, r);
        // 2.
        r.CopyDeviceTo(z);
        VecMulExecute(z, di_inv);

        r.CopyDeviceTo(mr);
        VecMulExecute(mr, di_inv);
        var mrr0 = SBlas.Dot(mr, r);

        int iter = 0;
        for (; iter < _maxIter; iter++)
        {
            // 3.
            z.CopyDeviceTo(az);
            matrix.Mul(z, az);

            var azz = SBlas.Dot(az, z);
            var alpha = mrr0 / azz;
            // 4.
            SBlas.Axpy(alpha, z, _x);
            // 5.
            SBlas.Axpy(-alpha, az, r);
            // 6.
            r.CopyDeviceTo(mr);
            VecMulExecute(di_inv, r);
            var mrr1 = SBlas.Dot(mr, r);
            var beta = mrr1/mrr0;
            // 7.
            SBlas.Scale(beta, z);
            SBlas.Axpy(1, mr, z);

            mrr0 = mrr1;

            var rr = SBlas.Dot(r, r);
            var bb = SBlas.Dot(_b, _b);
            if (rr / bb < _eps)
            {
                break;
            }
        }

        matrix.Mul(_x, z);
        _b.CopyDeviceTo(r);
        SBlas.Axpy(-1, z, r);
        // BLAS.axpy(_x.Length, -1, t, r);
        var rr2 = SBlas.Dot(r, r);
        _x.DeviceReadTo(x);

        return (rr2, iter);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: освободить управляемое состояние (управляемые объекты)
            }

            // TODO: надо вернуть обратно
            // r.Dispose();
            // r_hat.Dispose();
            // p.Dispose();
            // nu.Dispose();
            // h.Dispose();
            // s.Dispose();
            // t.Dispose();
            // dotpart.Dispose();
            // dotres.Dispose();
            // TODO: освободить неуправляемые ресурсы (неуправляемые объекты) и переопределить метод завершения
            // TODO: установить значение NULL для больших полей
            disposedValue = true;
        }
    }

    // // TODO: переопределить метод завершения, только если "Dispose(bool disposing)" содержит код для освобождения неуправляемых ресурсов
    ~CgmOCL()
    {
        // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
        Dispose(disposing: false);
    }

    public void Dispose()
    {
        // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
