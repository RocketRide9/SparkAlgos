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
