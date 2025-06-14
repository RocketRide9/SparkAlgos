using Real = double;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public ref struct SymDiagMatrixRef : MatrixContainer
{
    public required Span<Real> d3;
    public required Span<Real> d2;
    public required Span<Real> d1;
    public required Span<Real> d0;
    public required Span<Real> Di;

    public required int Gap;
    
    public int Size => Di.Length;
}


public class SymDiagMatrix : Matrix
{
    // diagonal
    public ComputeBuffer<Real> d3;
    public ComputeBuffer<Real> d2;
    public ComputeBuffer<Real> d1;
    public ComputeBuffer<Real> d0;
    // Основная диагональ
    public ComputeBuffer<Real> Di;

    // d0 находятся "вплотную" к основной диагонали
    // d1, d2, d3 находятся стоят "вплотную" друг к другу
    // d1 смещена на Gap элементов от d0.
    // Например, если они находятся вплотную друг к друг,
    // то Gap == 1.
    public int Gap;

    public int Size => Di.Length;
    ComputeBuffer<Real> Matrix.Di => Di;

    static SparkCL.Kernel? kernMul;

    public SymDiagMatrix(SymDiagMatrixRef matrix)
    {
        d3 = new ComputeBuffer<Real>(matrix.d3, BufferFlags.OnDevice);
        d2 = new ComputeBuffer<Real>(matrix.d2, BufferFlags.OnDevice);
        d1 = new ComputeBuffer<Real>(matrix.d1, BufferFlags.OnDevice);
        d0 = new ComputeBuffer<Real>(matrix.d0, BufferFlags.OnDevice);
        Di = new ComputeBuffer<Real>(matrix.Di, BufferFlags.OnDevice);
    
        Gap = matrix.Gap;
    }

    public void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernMul == null)
        {
            var support = new ComputeProgram("Matrices/SymDiagMatrix.cl");
            var localWork = new NDRange(Core.Prefered1D);

            kernMul = support.GetKernel(
                "SymDiagMul",
                new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D),
                localWork
            );
        }
            kernMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D);
            kernMul.SetArg(0, d3);
            kernMul.SetArg(1, d2);
            kernMul.SetArg(2, d1);
            kernMul.SetArg(3, d0);
            kernMul.SetArg(4, Di);
            kernMul.SetArg(5, vec.Length);
            kernMul.SetArg(6, Gap);

        kernMul.SetArg(7, vec);
        kernMul.SetArg(8, res);

        kernMul.Execute();
    }
}
