using Real = double;

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

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

    int Matrix.Size => Di.Length;
    ComputeBuffer<double> Matrix.Di => Di;

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

    public void Mul(ComputeBuffer<double> vec, ComputeBuffer<double> res)
    {
        throw new NotImplementedException();
        #if false
        if (kernMul == null)
        {
            var support = new ComputeProgram("Matrices/DiagMatrix.cl");
            var localWork = new NDRange(32);

            kernMul = support.GetKernel(
                "DIAGMul",
                new(NDRange.PaddedTo(vec.Length, 32)),
                localWork
            );
            kernMul.SetArg(0, Elems);
            kernMul.SetArg(1, Di);
            kernMul.SetArg(2, Ia);
            kernMul.SetArg(3, Ja);
            kernMul.SetArg(4, vec.Length);
        }

        kernMul.SetArg(5, vec);
        kernMul.SetArg(6, res);

        kernMul.Execute();
        #endif
    }
}
