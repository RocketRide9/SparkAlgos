using Real = double;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public ref struct MsrMatrixRef : MatrixContainer
{
    public required Span<Real> Elems;
    public required Span<Real> Di;
    public required Span<int> Ia;
    public required Span<int> Ja;

    public int Size => Di.Length;
}

public class MsrMatrix : Matrix
{
    public ComputeBuffer<Real> Elems;
    public ComputeBuffer<Real> Di;
    public ComputeBuffer<int> Ia;
    public ComputeBuffer<int> Ja;

    public int Size => Di.Length;
    ComputeBuffer<Real> Matrix.Di => Di;

    static SparkCL.Kernel? kernMul;
    public MsrMatrix(MsrMatrixRef matrix)
    {
        Elems = new ComputeBuffer<Real> (matrix.Elems, BufferFlags.OnDevice);
        Ia    = new ComputeBuffer<int>  (matrix.Ia, BufferFlags.OnDevice);
        Ja    = new ComputeBuffer<int>  (matrix.Ja, BufferFlags.OnDevice);
        Di    = new ComputeBuffer<Real> (matrix.Di, BufferFlags.OnDevice);
    }

    public void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernMul == null)
        {
            var support = new ComputeProgram("Matrices/MsrMatrix.cl");
            var localWork = new NDRange(Core.Prefered1D);

            kernMul = support.GetKernel(
                "MsrMul",
                new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D),
                localWork
            );
        }
            kernMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D);
            kernMul.SetArg(0, Elems);
            kernMul.SetArg(1, Di);
            kernMul.SetArg(2, Ia);
            kernMul.SetArg(3, Ja);
            kernMul.SetArg(4, vec.Length);

        kernMul.SetArg(5, vec);
        kernMul.SetArg(6, res);

        kernMul.Execute();
    }
}
