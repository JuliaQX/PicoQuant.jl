module CUDATests

using Test
using HDF5
using FFTW

using PicoQuant
include("test_utils.jl")

using CUDA
try
    device = CUDA.device()

    @testset "CUDA tests" begin
        @testset "Test state preparation algorithm" begin
            @test begin
                qubits = 3
                circ = create_simple_preparation_circuit(qubits, 4, 42)
                
                ψ′ = CuArray(get_statevector_using_qiskit(circ, big_endian=true))


                tng = convert_qiskit_circ_to_network(circ, InteractiveBackend{CuArray{ComplexF64}}())
                add_input!(tng, "000")
                out_node = full_wavefunction_contraction!(tng, "vector")

                ψ = load_tensor_data(tng, out_node)

                overlap = (a, b) -> abs(transpose(conj.(a)) * b)

                overlap(ψ, ψ′) ≈ 1.
            end


            @test begin
                qubits = 3
                circ = create_simple_preparation_circuit(qubits, 4, 42)
                
                ψ′ = CuArray(get_statevector_using_qiskit(circ, big_endian=true))

                tng = convert_qiskit_circ_to_network(circ, InteractiveBackend{CuArray{ComplexF64}}())
                add_input!(tng, "000")
                mps_nodes = contract_mps_tensor_network_circuit!(tng)
                calculate_mps_amplitudes!(tng, mps_nodes)

                ψ = load_tensor_data(tng, :result)

                overlap = (a, b) -> abs(transpose(conj.(a)) * b)

                overlap(ψ, ψ′) ≈ 1.
            end
        end
    end
catch
    println("Skipping CUDA tests as CUDA.jl failed to initialize")
end

end