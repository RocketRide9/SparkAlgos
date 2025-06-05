using Real = double;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public class MsrMatrix : Matrix
{
    public ComputeBuffer<Real> Elems;
    public ComputeBuffer<Real> Di;
    public ComputeBuffer<int> Ia;
    public ComputeBuffer<int> Ja;

    int Matrix.Size => Di.Length;
    ComputeBuffer<double> Matrix.Di => Di;

    static SparkCL.Kernel? kernMul;
    public MsrMatrix(MsrMatrixRef matrix)
    {
        Elems = new ComputeBuffer<Real> (matrix.Elems, BufferFlags.OnDevice);
        Ia    = new ComputeBuffer<int>  (matrix.Ia, BufferFlags.OnDevice);
        Ja    = new ComputeBuffer<int>  (matrix.Ja, BufferFlags.OnDevice);
        Di    = new ComputeBuffer<Real> (matrix.Di, BufferFlags.OnDevice);
    }

    public void Mul(ComputeBuffer<double> vec, ComputeBuffer<double> res)
    {
        if (kernMul == null)
        {
            var support = new ComputeProgram("Matrices/MsrMatrix.cl");
            var localWork = new NDRange(32);

            kernMul = support.GetKernel(
                "MSRMul",
                new NDRange((nuint)vec.Length).PadTo(32),
                localWork
            );
        }
            kernMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(32);
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
