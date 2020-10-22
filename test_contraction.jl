using TensorOperations
using CUDA

let
    n = 1 << 12
    # test a common operation when contracting two tensors
    a = CUDA.rand(ComplexF32, n, 2, n)
    b = CUDA.rand(ComplexF32, n, 2, n)

    a_indices = Tuple([-1, -2, 1])
    b_indices = Tuple([1, -3, -4])
    c_indices = Tuple(symdiff(a_indices, b_indices))

    CUDA.@profile c = tensorcontract(a, a_indices, b,b_indices, c_indices)

    @show typeof(c)
    @show size(c)
end
