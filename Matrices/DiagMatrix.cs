using Real = double;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public ref struct DiagMatrixRef : MatrixContainer
{
    // Left diagonal
    public required Span<Real> Ld3;
    public required Span<Real> Ld2;
    public required Span<Real> Ld1;
    public required Span<Real> Ld0;
    // Основная диагональ
    public required Span<Real> Di;
    // Right diagonal
    public required Span<Real> Rd0;
    public required Span<Real> Rd1;
    public required Span<Real> Rd2;
    public required Span<Real> Rd3;

    // Ld0 и Rd0 (*d0) находятся "вплотную" к основной диагонали
    // *d1, *d2, *d3 находятся стоят "вплотную" друг к другу
    // *d1 смещена на Gap элементов от *d0.
    // Например, если они находятся вплотную друг к друг,
    // то Gap == 1.
    public required int Gap;
    
    public int Size => Di.Length;
}

public class DiagMatrix : Matrix
{
    // Left diagonal
    public ComputeBuffer<Real> Ld3;
    public ComputeBuffer<Real> Ld2;
    public ComputeBuffer<Real> Ld1;
    public ComputeBuffer<Real> Ld0;
    // Основная диагональ
    public ComputeBuffer<Real> Di;
    // Right diagonal
    public ComputeBuffer<Real> Rd0;
    public ComputeBuffer<Real> Rd1;
    public ComputeBuffer<Real> Rd2;
    public ComputeBuffer<Real> Rd3;

    // Ld0 и Rd0 (*d0) находятся "вплотную" к основной диагонали
    // *d1, *d2, *d3 находятся стоят "вплотную" друг к другу
    // *d1 смещена на Gap элементов от *d0.
    // Например, если они находятся вплотную друг к друг,
    // то Gap == 1.
    public int Gap;

    public int Size => Di.Length;
    ComputeBuffer<Real> Matrix.Di => Di;

    static SparkCL.Kernel? kernMul;

    public DiagMatrix(DiagMatrixRef matrix)
    {
        Ld3 = new ComputeBuffer<Real>(matrix.Ld3, BufferFlags.OnDevice);
        Ld2 = new ComputeBuffer<Real>(matrix.Ld2, BufferFlags.OnDevice);
        Ld1 = new ComputeBuffer<Real>(matrix.Ld1, BufferFlags.OnDevice);
        Ld0 = new ComputeBuffer<Real>(matrix.Ld0, BufferFlags.OnDevice);
        Di = new ComputeBuffer<Real>(matrix.Di, BufferFlags.OnDevice);
        Rd0 = new ComputeBuffer<Real>(matrix.Rd0, BufferFlags.OnDevice);
        Rd1 = new ComputeBuffer<Real>(matrix.Rd1, BufferFlags.OnDevice);
        Rd2 = new ComputeBuffer<Real>(matrix.Rd2, BufferFlags.OnDevice);
        Rd3 = new ComputeBuffer<Real>(matrix.Rd3, BufferFlags.OnDevice);
    
        Gap = matrix.Gap;
    }

    public void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernMul == null)
        {
            var support = new ComputeProgram("Matrices/DiagMatrix.cl");
            var localWork = new NDRange(32);

            kernMul = support.GetKernel(
                "DiagMul",
                new NDRange((nuint)vec.Length).PadTo(32),
                localWork
            );
        }
            kernMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(32);
            kernMul.SetArg(0, Ld3);
            kernMul.SetArg(1, Ld2);
            kernMul.SetArg(2, Ld1);
            kernMul.SetArg(3, Ld0);
            kernMul.SetArg(4, Di);
            kernMul.SetArg(5, Rd0);
            kernMul.SetArg(6, Rd1);
            kernMul.SetArg(7, Rd2);
            kernMul.SetArg(8, Rd3);
            kernMul.SetArg(9, vec.Length);
            kernMul.SetArg(10, Gap);

        kernMul.SetArg(11, vec);
        kernMul.SetArg(12, res);

        kernMul.Execute();
    }
}
