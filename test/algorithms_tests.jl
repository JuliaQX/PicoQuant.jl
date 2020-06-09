using FFTW
using PyCall

# *************************************************************************** #
#           Some utility functions to help with testing
# *************************************************************************** #

"""
    function switch_endianness(vec)

Reverses the qubit ordering for a given state vector
"""
function switch_endianness(vec)
    n = convert(Int, log2(length(vec)))
    vec = reshape(vec, Tuple([2 for _ in 1:n]))
    vec = permutedims(vec, [i for i in n:-1:1])
    reshape(vec, 2^n)
end

"""
    function get_statevector_using_qiskit(circ; big_endian=true)

Uses qiskit to calculate the state vector for a given circuit
"""
function get_statevector_using_qiskit(circ; big_endian=true)
    qiskit = pyimport("qiskit")
    backend = qiskit.Aer.get_backend("statevector_simulator")
    job = qiskit.execute(circ, backend)
    result = job.result()
    vec = result.get_statevector()
    if !big_endian
        return switch_endianness(vec)
    else
        return vec
    end
end

"""
    function get_statevector_using_picoquant(circ; big_endian=false)

Uses picoquant to calculate statevector resulting from given circuit
"""
function get_statevector_using_picoquant(circ; big_endian=false)
    InteractiveBackend()
    tn = convert_qiskit_circ_to_network(circ, decompose=false, transpile=false)
    qubits = circ.n_qubits
    add_input!(tn, "0"^qubits)
    plan = inorder_contraction_plan(tn)
    contract_network!(tn, plan)
    node = iterate(values(tn.nodes))[1]
    idx_order = [findfirst(x -> x == i, node.indices) for i in tn.output_qubits]
    if !big_endian
        idx_order = idx_order[end:-1:1]
    end
    reshape(permutedims(backend.tensors[:result], idx_order), 2^qubits)
end


# *************************************************************************** #
#           Testing
# *************************************************************************** #

@testset "Test state preparation algorithm" begin
    @test begin
        qubits = 1
        circ = create_simple_preparation_circuit(qubits, 4, 42)


        ψ = get_statevector_using_picoquant(circ,
                                            big_endian=false)
        ψ′ = get_statevector_using_qiskit(circ, big_endian=false)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ψ′) ≈ 1.
    end

    @test begin
        qubits = 3
        circ = create_simple_preparation_circuit(qubits, 4, 42)

        ψ = get_statevector_using_picoquant(circ,
                                            big_endian=false)
        ψ′ = get_statevector_using_qiskit(circ, big_endian=false)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ψ′) ≈ 1.
    end
end

@testset "Small QFT circuit test" begin
    @test begin
        n = 3

        prep_circ = create_simple_preparation_circuit(n, 3, 43)
        qft_circ = create_qft_circuit(n)

        full_circ = prep_circ.combine(qft_circ)

        qft_input = get_statevector_using_qiskit(prep_circ, big_endian=false)
        ref_output = ifft(qft_input)
        norm = x -> x ./ sqrt(sum(x .* conj.(x)))
        ref_output = norm(ref_output)

        ψ = get_statevector_using_picoquant(full_circ)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ref_output) ≈ 1.
    end
end


@testset "Larger QFT circuit test" begin
    @test begin
        n = 8

        prep_circ = create_simple_preparation_circuit(n, 3, 43)
        qft_circ = create_qft_circuit(n)

        full_circ = prep_circ.combine(qft_circ)

        qft_input = get_statevector_using_qiskit(prep_circ, big_endian=false)
        ref_output = ifft(qft_input)
        norm = x -> x ./ sqrt(sum(x .* conj.(x)))
        ref_output = norm(ref_output)

        ψ = get_statevector_using_picoquant(full_circ)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ref_output) ≈ 1.
    end
end
