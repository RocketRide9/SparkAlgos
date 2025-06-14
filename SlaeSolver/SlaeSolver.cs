using Real = double;

namespace SparkAlgos.SlaeSolver;

public interface ISlaeSolver
{
    static abstract ISlaeSolver Construct(int maxIter, Real eps);
    // x используется как начальное приближение, туда же попадёт ответ
    (Real discrep, int iter) Solve(Types.Matrix matrix, Span<Real> b, Span<Real> x);
    void AllocateTemps(int n);
}
