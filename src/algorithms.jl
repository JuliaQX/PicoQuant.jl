using Random

export create_qft_circuit, create_simple_preparation_circuit
export create_ghz_preparation_circuit

"""
    function create_qft_circuit(qubits::Integer)

Generate QFT circuit acting on given number of qubits
"""
function create_qft_circuit(qubits::Integer)
    qiskit = pyimport("qiskit")
    circ = qiskit.QuantumCircuit(qubits)
    for i in 1:qubits
        circ.h(i-1)
        for j in i:qubits-1
            circ.cu1(π/(2^(j-i+1)), j, i-1)
        end
        circ.barrier()
    end
    for i = 1:convert(Integer, floor(qubits//2))
        circ.swap(i - 1, qubits - i)
    end
    circ
end

"""
    function create_simple_preparation_circuit(qubits::Integer,
                                               depth::Integer,
                                               seed::Integer=nothing)

Create a simple preparation circuit which mixes
"""
function create_simple_preparation_circuit(qubits::Integer,
                                           depth::Integer,
                                           seed::Union{Integer, Nothing}=nothing)
    if seed == nothing
        rng = MersenneTwister()
    else
        rng = MersenneTwister(seed)
    end

    qiskit = pyimport("qiskit")
    circ = qiskit.QuantumCircuit(qubits)

    # add layer of Hadamard gates
    for q in 1:qubits
        circ.h(q - 1)
    end
    circ.barrier()

    # add layers of random unitaries and contolled not gates
    for d in 1:depth
        params = rand(rng, qubits, 3)
        for q in 1:qubits
            circ.u3(params[q,:]..., q - 1)
        end
        start_qubit = ((d + 1) % 2) + 1
        for q in start_qubit:2:qubits - 1
            circ.cx(q - 1, q)
        end
        circ.barrier()
    end
    circ
end

"""
    function create_ghz_preparation_circuit(qubits::Integer)

Create a ghz preparation circuit
"""
function create_ghz_preparation_circuit(qubits::Integer)
    qiskit = pyimport("qiskit")
    circ = qiskit.QuantumCircuit(qubits)

    circ.h(0)
    for q in 1:qubits-1
        circ.cx(q-1, q)
    end

    circ
end
