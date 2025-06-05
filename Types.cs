using Real = double;

using SparkCL;

namespace SparkAlgos.Types;

public interface MatrixContainer
{
    int Size { get; }
};

public interface Matrix
{
    int Size { get; }
    // TODO: нужно для предобуславливания.
    // Надо придумать что-то более разумное
    ComputeBuffer<Real> Di { get; }

    void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res);
}

public ref struct MsrMatrixRef : MatrixContainer
{
    public required Span<Real> Elems;
    public required Span<Real> Di;
    public required Span<int> Ia;
    public required Span<int> Ja;

    public int Size => Di.Length;
}

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
