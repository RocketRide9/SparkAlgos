using SparkCL;
using SparkOCL;
using Real = float;

namespace SparkAlgos;

public class BicgStab : IDisposable
{
    SparkCL.Memory<Real> _mat;
    SparkCL.Memory<Real> _di;
    SparkCL.Memory<Real> _b;
    SparkCL.Memory<int> _ia;
    SparkCL.Memory<int> _ja;

    int _maxIter;
    Real _eps;
    SparkCL.Memory<Real> _x;
    
    SparkCL.Memory<Real> r;
    SparkCL.Memory<Real> di_inv;
    SparkCL.Memory<Real> y;
    SparkCL.Memory<Real> z;
    SparkCL.Memory<Real> ks;
    SparkCL.Memory<Real> kt;
    SparkCL.Memory<Real> r_hat;
    SparkCL.Memory<Real> p;
    SparkCL.Memory<Real> nu;
    SparkCL.Memory<Real> h;
    SparkCL.Memory<Real> s;
    SparkCL.Memory<Real> t;
    SparkCL.Memory<Real> dotpart;
    SparkCL.Memory<Real> dotres;
    private bool disposedValue;

    public BicgStab(
        SparkOCL.Array<Real> Mat,
        SparkOCL.Array<Real> Di,
        SparkOCL.Array<Real> B,
        SparkOCL.Array<int> Ia,
        SparkOCL.Array<int> Ja,

        SparkOCL.Array<Real> x0,
        int maxIter,
        Real eps)
    {
        _maxIter = maxIter;
        _eps = eps;

        _mat = SparkCL.Memory<Real>.ForArray(Mat); 
        _di = SparkCL.Memory<Real>.ForArray(Di); 
        _b = SparkCL.Memory<Real>.ForArray(B); 
        _ia = SparkCL.Memory<int>.ForArray(Ia); 
        _ja = SparkCL.Memory<int>.ForArray(Ja); 

        _x = SparkCL.Memory<Real>.ForArray(x0);
        _x.Write();
        _b.Write();
        _ia.Write();
        _ja.Write();
        _mat.Write();
        _di.Write();

        r       = new SparkCL.Memory<Real>(_b.Count);
        r_hat   = new SparkCL.Memory<Real>(_b.Count);
        p       = new SparkCL.Memory<Real>(_b.Count);
        nu      = new SparkCL.Memory<Real>(_b.Count);
        h       = new SparkCL.Memory<Real>(_b.Count);
        s       = new SparkCL.Memory<Real>(_b.Count);
        t       = new SparkCL.Memory<Real>(_b.Count);
        di_inv  = new SparkCL.Memory<Real>(_b.Count);
        y       = new SparkCL.Memory<Real>(_b.Count);
        z       = new SparkCL.Memory<Real>(_b.Count);
        ks      = new SparkCL.Memory<Real>(_b.Count);
        kt      = new SparkCL.Memory<Real>(_b.Count);
        dotpart = new SparkCL.Memory<Real>(32*2);
        dotres  = new SparkCL.Memory<Real>(1);
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

    public (Array<Real> ans, Real rr, Real pp, int iter) Solve()
    {
        var globalWork = new NDRange(PaddedTo(_x.Count, 32));
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
            kernDiscrep.PushArg(_x.Count);
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
            kernP.SetArg(5, p.Count);

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
            kernMul.SetArg(4, _x.Count);
        
        Event MulExecute(SparkCL.Memory<Real> _a, SparkCL.Memory<Real> _res){
            kernMul.SetArg(5, _a);
            kernMul.SetArg(6, _res);
            return kernMul.Execute();
        }
        
        var kernRsqrt = solvers.GetKernel(
            "BLAS_rsqrt",
            globalWork,
            localWork
        );
        Event RsqrtExecute(SparkCL.Memory<Real> _y) {
            kernRsqrt.SetArg(0, _y);
            kernRsqrt.SetArg(1, _y.Count);
            return kernRsqrt.Execute();
        }
        
        var kernVecMul = solvers.GetKernel(
            "VecMul",
            globalWork,
            localWork
        );
        Event VecMulExecute(SparkCL.Memory<Real> _y, SparkCL.Memory<Real> _x) {
            kernVecMul.SetArg(0, _y);
            kernVecMul.SetArg(1, _x);
            kernVecMul.SetArg(2, _y.Count);
            return kernVecMul.Execute();
        }
        
        var SBlas = SparkAlgos.Blas.GetInstance();
        SBlas.Scratch64 = dotpart;
        SBlas.Scratch1 = dotres;

        // precond
        _di.CopyTo(di_inv);
        RsqrtExecute(di_inv);
        // BiCGSTAB
        // 1.
        kernDiscrep.Execute();
        // 2.
        r.CopyTo(r_hat);
        // 3.
        Real pp = SBlas.Dot(r, r); // r_hat * r
        // 4.
        r.CopyTo(p);

        int iter = 0;
        Real rr;
        for (; iter < _maxIter; iter++)
        {
            // 1.
            p.CopyTo(y);
            VecMulExecute(y, di_inv);
            VecMulExecute(y, di_inv);
            // 2.
            MulExecute(y, nu);
            
            // 3.
            Real rnu = SBlas.Dot(r_hat, nu);
            Real alpha = pp / rnu;

            // 4. h = x + alpha*p
            _x.CopyTo(h);
            SBlas.Axpy(alpha, y, h);
            
            // 5.
            r.CopyTo(s);
            SBlas.Axpy(-alpha, nu, s);

            // 6.
            Real ss = SBlas.Dot(s, s);
            if (ss < _eps)
            {
                // тогда h - решение
                h.CopyTo(_x);
                // тогда h - решение. Предыдущий вектор x можно освободить
                //_x.Dispose();
                //_x = h;
                break;
            }
            
            // 7.
            s.CopyTo(ks);
            VecMulExecute(ks, di_inv);
            ks.CopyTo(z);
            VecMulExecute(z, di_inv);

            // 8.
            MulExecute(z, t);

            // 9.
            t.CopyTo(kt);
            VecMulExecute(kt, di_inv);
            
            Real ts = SBlas.Dot(ks, kt);
            Real tt = SBlas.Dot(kt, kt);
            Real w = ts / tt;

            // 10. 
            h.CopyTo(_x);
            SBlas.Axpy(w, z, _x);

            // 11.
            s.CopyTo(r);
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
            PExecute(w, beta);

            pp = pp1;
        }

        // get the true discrepancy
        kernDiscrep.Execute();
        rr = SBlas.Dot(r, r);

        _x.Read();
        return (_x.Array, rr, pp, iter);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: освободить управляемое состояние (управляемые объекты)
            }

            r.Dispose();
            r_hat.Dispose();
            p.Dispose();
            nu.Dispose();
            h.Dispose();
            s.Dispose();
            t.Dispose();
            dotpart.Dispose();
            dotres.Dispose();
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
