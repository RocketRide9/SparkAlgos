using Silk.NET.OpenCL;
using SparkCL;
using OCLHelper;
using Real = double;

namespace SparkAlgos;

public ref struct SlaeRef
{
    public Span<Real> Mat;
    public Span<Real> Di;
    public Span<Real> B;
    public Span<int> Ia;
    public Span<int> Ja;
}

public class Blas
{
    static Blas? instance = null;

    public static Blas GetInstance()
    {
        if (instance == null) instance = new Blas();

        return instance;
    }

    SparkCL.Kernel _dot1;
    SparkCL.Kernel _dot2;
    SparkCL.Kernel _scale;
    SparkCL.Kernel _axpy;
    SparkCL.Program _solvers;

    // probably not the most elegant solution, but it
    // avoid hidden allocation and passing scratch buffers
    // via paramenters
    // number means the minimum required length
    public ComputeBuffer<Real>? Scratch64 { get; set; }
    public ComputeBuffer<Real>? Scratch1 { get; set;}

    private Blas()
    {
        var localWork = new OCLHelper.NDRange(16);

        _solvers = new SparkCL.Program("Blas.cl");
        _dot1 = _solvers.GetKernel(
            "Xdot",
            globalWork: new(32*32*2),
            localWork: new(32)
        );
        _dot2 = _solvers.GetKernel(
            "XdotEpilogue",
            globalWork: new(32),
            localWork: new(32)
        );
        // global work: not nice
        _scale = _solvers.GetKernel(
            "BLAS_scale",
            globalWork: new(1),
            localWork: localWork
        );
        _axpy = _solvers.GetKernel(
            "BLAS_axpy",
            globalWork: new(1),
            localWork: localWork
        );
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

    /// requires scratch64 and scratch1
    public Real Dot(SparkCL.ComputeBuffer<Real> x, SparkCL.ComputeBuffer<Real> y)
    {
        _dot1.SetArg(0, x.Length);
        _dot1.SetArg(1, x);
        _dot1.SetArg(2, y);
        _dot1.SetArg(3, Scratch64!);
        _dot1.Execute();

        _dot2.SetArg(0, Scratch64!);
        _dot2.SetArg(1, Scratch1!);
        _dot2.Execute();

        Span<Real> flt_span = stackalloc Real[1];
        Scratch1.DeviceReadTo(flt_span);

        return flt_span[0];
    }

    public void Scale(Real a, SparkCL.ComputeBuffer<Real> y)
    {
        _scale.GlobalWork = new(PaddedTo(y.Length, 32));
        _scale.SetArg(0, a);
        _scale.SetArg(1, y);
        _scale.SetArg(2, y.Length);

        _scale.Execute();
    }

    public void Axpy(Real a, SparkCL.ComputeBuffer<Real> x, SparkCL.ComputeBuffer<Real> y)
    {
        _axpy.GlobalWork = new(PaddedTo(y.Length, 32));
        _axpy.SetArg(0, a);
        _axpy.SetArg(1, x);
        _axpy.SetArg(2, y);
        _axpy.SetArg(3, y.Length);

        _axpy.Execute();
    }
}
