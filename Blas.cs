using SparkCL;
using Real = float;

namespace SparkAlgos;

public class Blas
{
    static Blas? instance = null;

    public static Blas GetInstance()
    {
        if (instance == null) instance = new Blas();
        
        return instance;
    }

    Kernel _dot1;
    Kernel _dot2;
    Kernel _scale;
    Kernel _axpy;
    SparkCL.Program _solvers;

    // probably not the most elegant solution, but it
    // avoid hidden allocation and passing scratch buffers
    // via paramenters
    // number means the minimum required length
    public SparkCL.Memory<Real>? Scratch64 { get; set; }
    public SparkCL.Memory<Real>? Scratch1 { get; set;}

    private Blas()
    {
        var localWork = new SparkOCL.NDRange(16);
        
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
    public Real Dot(SparkCL.Memory<Real> x, SparkCL.Memory<Real> y)
    {
        _dot1.SetArg(0, x.Count);
        _dot1.SetArg(1, x);
        _dot1.SetArg(2, y);
        _dot1.SetArg(3, Scratch64!);
        _dot1.Execute();

        _dot2.SetArg(0, Scratch64!);
        _dot2.SetArg(1, Scratch1!);
        _dot2.Execute();
        Scratch1!.Read(true);

        return Scratch1![0];
    }
    
    public void Scale(Real a, SparkCL.Memory<Real> y)
    {
        _scale.GlobalWork = new(PaddedTo(y.Count, 32));
        _scale.SetArg(0, a);
        _scale.SetArg(1, y);
        _scale.SetArg(2, y.Count);

        _scale.Execute();
    }
    
    public void Axpy(Real a, SparkCL.Memory<Real> x, SparkCL.Memory<Real> y)
    {
        _axpy.GlobalWork = new(PaddedTo(y.Count, 32));
        _axpy.SetArg(0, a);
        _axpy.SetArg(1, x);
        _axpy.SetArg(2, y);
        _axpy.SetArg(3, y.Count);

        _axpy.Execute();
    }
}
