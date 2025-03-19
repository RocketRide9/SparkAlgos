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
    const Real _eps = 1e-13f;
    public SparkCL.Memory<Real> X { get; private set; }
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
        SparkCL.Memory<Real> Mat,
        SparkCL.Memory<Real> Di,
        SparkCL.Memory<Real> B,
        SparkCL.Memory<int> Ia,
        SparkCL.Memory<int> Ja,

        SparkCL.Memory<Real> x0,
        int maxIter)
    {
        _maxIter = maxIter;

        _mat = Mat; 
        _di = Di; 
        _b = B; 
        _ia = Ia; 
        _ja = Ja; 

        X = x0;
        X.Write();
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

    public (Real rr, Real pp, int iter) Solve()
    {
        // Вынос векторов в текущую область видимости

        // BiCGSTAB
        var solvers = new SparkCL.Program("Solvers.cl");

        var prepare1 = solvers.GetKernel(
            "BiCGSTAB_prepare1",
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
        );
            prepare1.PushArg(_mat);
            prepare1.PushArg(_di);
            prepare1.PushArg(_ia);
            prepare1.PushArg(_ja);
            prepare1.PushArg(X.Count);
            prepare1.PushArg(r);
            prepare1.PushArg(_b);
            prepare1.PushArg(X);

        var kernP = solvers.GetKernel(
            "BiCGSTAB_p",
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
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
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
        );
            kernMul.SetArg(0, _mat);
            kernMul.SetArg(1, _di);
            kernMul.SetArg(2, _ia);
            kernMul.SetArg(3, _ja);
            kernMul.SetArg(4, X.Count);
        
        Event MulExecute(SparkCL.Memory<Real> _a, SparkCL.Memory<Real> _res){
            kernMul.SetArg(5, _a);
            kernMul.SetArg(6, _res);
            return kernMul.Execute();
        }
            
        var kernAxpy = solvers.GetKernel(
            "BLAS_axpy",
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
        );
        Event AxpyExecute(Real _a, SparkCL.Memory<Real> _x, SparkCL.Memory<Real> _y) {
            kernAxpy.SetArg(0, _a);
            kernAxpy.SetArg(1, _x);
            kernAxpy.SetArg(2, _y);
            kernAxpy.SetArg(3, _y.Count);
            return kernAxpy.Execute();
        }
        
        var kernRsqrt = solvers.GetKernel(
            "BLAS_rsqrt",
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
        );
        Event RsqrtExecute(SparkCL.Memory<Real> _y) {
            kernRsqrt.SetArg(0, _y);
            kernRsqrt.SetArg(1, _y.Count);
            return kernRsqrt.Execute();
        }
        
        var kernVecMul = solvers.GetKernel(
            "VecMul",
            globalWork: new(PaddedTo(X.Count, 32)),
            localWork:  new(32)
        );
        Event VecMulExecute(SparkCL.Memory<Real> _y, SparkCL.Memory<Real> _x) {
            kernVecMul.SetArg(0, _y);
            kernVecMul.SetArg(1, _x);
            kernVecMul.SetArg(2, _y.Count);
            return kernVecMul.Execute();
        }
        
        var kern1 = solvers.GetKernel(
            "Xdot",
            globalWork: new(32*32*2),
            localWork: new(32)
        );
        var kern2 = solvers.GetKernel(
            "XdotEpilogue",
            globalWork: new(32),
            localWork: new(32)
        );
        Real DotExecute(SparkCL.Memory<Real> _x, SparkCL.Memory<Real> _y)
        {
            kern1.SetArg(0, _x.Count);
            kern1.SetArg(1, _x);
            kern1.SetArg(2, _y);
            kern1.SetArg(3, dotpart);
            kern1.Execute();

            kern2.SetArg(0, dotpart);
            kern2.SetArg(1, dotres);
            kern2.Execute();
            dotres.Read(true);

            return dotres[0];
        }

        // precond
        _di.CopyTo(di_inv);
        RsqrtExecute(di_inv);
        // BiCGSTAB
        // 1.
        prepare1.Execute();
        // 2.
        r.CopyTo(r_hat);
        // 3.
        Real pp = DotExecute(r, r); // r_hat * r
        // 4.
        r.CopyTo(p);

        int iter = 0;
        Real rr = 0;
        for (; iter < _maxIter; iter++)
        {
            // 1.
            p.CopyTo(y);
            VecMulExecute(y, di_inv);
            VecMulExecute(y, di_inv);
            // 2.
            MulExecute(y, nu);
            
            // 3.
            Real rnu = DotExecute(r_hat, nu);
            Real alpha = pp / rnu;

            // 4. h = x + alpha*p
            X.CopyTo(h);
            AxpyExecute(alpha, y, h);
            
            // 5.
            r.CopyTo(s);
            AxpyExecute(-alpha, nu, s);

            // 6.
            Real ss = DotExecute(s, s);
            if (ss < _eps)
            {
                // тогда h - решение. Предыдущий вектор x можно освободить
                X.Dispose();
                X = h;
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
            
            Real ts = DotExecute(ks, kt);
            Real tt = DotExecute(kt, kt);
            Real w = ts / tt;

            // 10. 
            h.CopyTo(X);
            AxpyExecute(w, z, X);

            // 11.
            s.CopyTo(r);
            AxpyExecute(-w, t, r);
            
            // 12.
            rr = DotExecute(r, r);
            if (rr < _eps)
            {
                break;
            }

            // 13-14.
            Real pp1 = DotExecute(r, r_hat);
            Real beta = (pp1 / pp) * (alpha / w);

            // 15.
            PExecute(w, beta);

            Core.WaitQueue();
            pp = pp1;
        }

        X.Read(true);
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
