using Silk.NET.OpenCL;
using SparkCL;
using OCLHelper;
using Real = double;

namespace SparkAlgos;

public class BicgStab : IDisposable
{
    int _maxIter;
    Real _eps;

    int _n = 0; // размерность СЛАУ
    ComputeBuffer<Real> r;
    ComputeBuffer<Real> di_inv;
    ComputeBuffer<Real> y;
    ComputeBuffer<Real> z;
    ComputeBuffer<Real> ks;
    ComputeBuffer<Real> kt;
    ComputeBuffer<Real> r_hat;
    ComputeBuffer<Real> p;
    ComputeBuffer<Real> nu;
    ComputeBuffer<Real> h;
    ComputeBuffer<Real> s;
    ComputeBuffer<Real> t;
    ComputeBuffer<Real> dotpart;
    ComputeBuffer<Real> dotres;
    private bool disposedValue;
    
    

    public BicgStab(
        int maxIter,
        Real eps
    ) {
        _maxIter = maxIter;
        _eps = eps;

        dotpart = new ComputeBuffer<Real>(32*2, BufferFlags.OnDevice);
        dotres  = new ComputeBuffer<Real>(1, BufferFlags.OnDevice);
    }

    static nuint PaddedTo(int initial, int multiplier)
    {
        if (initial % multiplier == 0)
        {
            return (nuint)initial;
        } else {
            return ((nuint)initial / 32 + 1 ) * 32;
        }
    }
    
    // Выделить память для временных массивов
    // n - длина каждого массива
    public void AllocateTemps(int n)
    {
        if (n != _n)
        {
            _n = n;

            r       = new (n, BufferFlags.OnDevice);
            r_hat   = new (n, BufferFlags.OnDevice);
            p       = new (n, BufferFlags.OnDevice);
            nu      = new (n, BufferFlags.OnDevice);
            h       = new (n, BufferFlags.OnDevice);
            s       = new (n, BufferFlags.OnDevice);
            t       = new (n, BufferFlags.OnDevice);
            di_inv  = new (n, BufferFlags.OnDevice);
            y       = new (n, BufferFlags.OnDevice);
            z       = new (n, BufferFlags.OnDevice);
            ks      = new (n, BufferFlags.OnDevice);
            kt      = new (n, BufferFlags.OnDevice);
        }
    }

    public (Real rr, Real pp, int iter) Solve(SlaeRef slae, Span<Real> x)
    {
        var _mat = new ComputeBuffer<Real>(slae.Mat, BufferFlags.OnDevice);
        var _di  = new ComputeBuffer<Real>(slae.Di,  BufferFlags.OnDevice);
        var _b   = new ComputeBuffer<Real>(slae.B,   BufferFlags.OnDevice);
        var _ia  = new ComputeBuffer<int> (slae.Ia,  BufferFlags.OnDevice);
        var _ja  = new ComputeBuffer<int> (slae.Ja,  BufferFlags.OnDevice);
        var _x   = new ComputeBuffer<Real>(x,        BufferFlags.OnDevice);
        
        var globalWork = new NDRange(PaddedTo(x.Length, 32));
        var localWork = new NDRange(16);

        // BiCGSTAB
        var solvers = new SparkCL.Program("Solvers.cl");

        var kernDiscrep = solvers.GetKernel(
            "BiCGSTAB_disc",
            globalWork,
            localWork
        );
            kernDiscrep.PushArg(_mat);
            kernDiscrep.PushArg(_di);
            kernDiscrep.PushArg(_ia);
            kernDiscrep.PushArg(_ja);
            kernDiscrep.PushArg(_x.Length);
            kernDiscrep.PushArg(r);
            kernDiscrep.PushArg(_b);
            kernDiscrep.PushArg(_x);

        var kernP = solvers.GetKernel(
            "BiCGSTAB_p",
            globalWork,
            localWork
        );
            kernP.SetArg(0, p);
            kernP.SetArg(1, r);
            kernP.SetArg(2, nu);
            kernP.SetArg(5, p.Length);

        Event PExecute(Real _w, Real _beta)
        {
            kernP.SetArg(3, _w);
            kernP.SetArg(4, _beta);
            return kernP.Execute();
        }

        var kernMul = solvers.GetKernel(
            "MSRMul",
            globalWork,
            localWork
        );
            kernMul.SetArg(0, _mat);
            kernMul.SetArg(1, _di);
            kernMul.SetArg(2, _ia);
            kernMul.SetArg(3, _ja);
            kernMul.SetArg(4, _x.Length);

        Event MulExecute(ComputeBuffer<Real> _a, ComputeBuffer<Real> _res){
            kernMul.SetArg(5, _a);
            kernMul.SetArg(6, _res);
            return kernMul.Execute();
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
            new NDRange(PaddedTo(_x.Length/4, 16)),
            new(16)
        );
        Event VecMulExecute(ComputeBuffer<Real> _y, ComputeBuffer<Real> _x) {
            kernVecMul.SetArg(0, _y);
            kernVecMul.SetArg(1, _x);
            kernVecMul.SetArg(2, _y.Length);
            return kernVecMul.Execute();
        }

        var SBlas = SparkAlgos.Blas.GetInstance();
        SBlas.Scratch64 = dotpart;
        SBlas.Scratch1 = dotres;

        // precond
        _di.CopyDeviceTo(di_inv);
        RsqrtExecute(di_inv);
        // BiCGSTAB
        // 1.
        kernDiscrep.Execute();
        // 2.
        r.CopyDeviceTo(r_hat);
        // 3.
        Real pp = SBlas.Dot(r, r); // r_hat * r
        // 4.
        r.CopyDeviceTo(p);

        int iter = 0;
        Real rr;
        for (; iter < _maxIter; iter++)
        {
            // 1.
            p.CopyDeviceTo(y);
            VecMulExecute(y, di_inv);
            VecMulExecute(y, di_inv);
            // 2.
            MulExecute(y, nu);

            // 3.
            Real rnu = SBlas.Dot(r_hat, nu);
            Real alpha = pp / rnu;

            // 4. h = x + alpha*p
            _x.CopyDeviceTo(h);
            SBlas.Axpy(alpha, y, h);

            // 5.
            r.CopyDeviceTo(s);
            SBlas.Axpy(-alpha, nu, s);

            // 6.
            Real ss = SBlas.Dot(s, s);
            if (ss < _eps)
            {
                // тогда h - решение
                h.CopyDeviceTo(_x);
                // тогда h - решение. Предыдущий вектор x можно освободить
                //_x.Dispose();
                //_x = h;
                break;
            }

            // 7.
            s.CopyDeviceTo(ks);
            VecMulExecute(ks, di_inv);
            ks.CopyDeviceTo(z);
            VecMulExecute(z, di_inv);

            // 8.
            MulExecute(z, t);

            // 9.
            t.CopyDeviceTo(kt);
            VecMulExecute(kt, di_inv);

            Real ts = SBlas.Dot(ks, kt);
            Real tt = SBlas.Dot(kt, kt);
            Real w = ts / tt;

            // 10.
            h.CopyDeviceTo(_x);
            SBlas.Axpy(w, z, _x);

            // 11.
            s.CopyDeviceTo(r);
            SBlas.Axpy(-w, t, r);

            // 12.
            rr = SBlas.Dot(r, r);
            if (rr < _eps)
            {
                break;
            }

            // 13-14.
            Real pp1 = SBlas.Dot(r, r_hat);
            Real beta = (pp1 / pp) * (alpha / w);

            // 15.
            // SBlas.Axpy(-w, nu, p);
            // SBlas.Scale(beta, p);
            // SBlas.Axpy(1, r, p);
            PExecute(w, beta);

            pp = pp1;
        }

        // get the true discrepancy
        kernDiscrep.Execute();
        rr = SBlas.Dot(r, r);
        _x.DeviceReadTo(x);

        return (rr, pp, iter);
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
    ~BicgStab()
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
