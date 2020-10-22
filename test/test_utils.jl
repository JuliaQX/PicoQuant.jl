using PyCall


# Some utility functions to help with testing

export switch_endianness, get_statevector_using_picoquant, get_statevector_using_qiskit

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
    tn = convert_qiskit_circ_to_network(circ, InteractiveBackend{Array{ComplexF64}}(), decompose=false, transpile=false)
    qubits = circ.n_qubits
    add_input!(tn, "0"^qubits)
    full_wavefunction_contraction!(tn)
    result_tensor = load_tensor_data(tn, :result)
    dims = size(result_tensor)
    if !big_endian
        index_order = length(dims):-1:1
        result_tensor = permutedims(result_tensor, index_order)
    end
    reshape(result_tensor, prod(dims))
end

