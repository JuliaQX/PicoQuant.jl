using PyCall
using Random

export create_qft_circuit, create_simple_preparation_circuit
export create_ghz_preparation_circuit
export create_RQC

"""
    function create_qft_circuit(n::Int, m::Int=-1)

Generate QFT circuit acting on n qubits with approximation parameter m. To no approximation
m is set to -1 which is the default value.
"""
function create_qft_circuit(n::Int, m::Int=-1)
    qiskit = pyimport("qiskit")
    circ = qiskit.QuantumCircuit(n)
    m = (m == -1) ? n : m # if m is -1 then do not truncate
    for i in 1:n
        circ.h(i-1)
        for j in i:n-1
            if j - i < m
                circ.cu1(π/(2^(j-i+1)), j, i-1)
            end
        end
        circ.barrier()
    end
    # TODO: do not add swap gates but return ordering information instead
    for i = 1:convert(Int, floor(n//2))
        circ.swap(i - 1, n - i)
    end
    circ
end

"""
    function create_simple_preparation_circuit(qubits::Int,
                                               depth::Int,
                                               seed::Int=nothing)

Create a simple preparation circuit which mixes
"""
function create_simple_preparation_circuit(qubits::Int,
                                           depth::Int,
                                           seed::Union{Int, Nothing}=nothing)
    if seed === nothing
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
    function create_ghz_preparation_circuit(qubits::Int)

Create a ghz preparation circuit
"""
function create_ghz_preparation_circuit(qubits::Int)
    qiskit = pyimport("qiskit")
    circ = qiskit.QuantumCircuit(qubits)

    circ.h(0)
    for q in 1:qubits-1
        circ.cx(q-1, q)
    end

    circ
end

# *************************************************************************** #
#                              RQC functions
# *************************************************************************** #

"Struct to represent a random quantum circuit"
struct RQC
    # The dimensions of the 2-dimensional array of qubits.
    n::Int
    m::Int

    # A qiskit circuit object for the R.Q.C.
    circ

    # A 2-dimensional array of integers to keep track of which single qubit
    # gate should next be applied to a given qubit in the circuit.
    # The integers map to gates as follows:
    # 1 => T, 2 => sqrt(X), 3 => sqrt(Y).
    # The sign of the integer is used to indicate if the corresponding qubit
    # has been hit by a two qubit gate and can be hit by a single qubit gate.
    # This should be initialised as an array of ones as the first single qubit
    # gate is always a T gate.
    next_gate::Array{Int, 2}

    # A dictionary to convert integers from the above array into gate names.
    single_qubit_gates::Dict{Int, String}
end

function RQC(n::Int, m::Int)
    qiskit = pyimport("qiskit")
    RQC(n, m, qiskit.QuantumCircuit(n*m), -ones(Int, n,m),
        Dict(1=>"t", 2=>"x", 3=>"y"))
end

"""
    function random_gate!(rqc::RQC, i::Int, j::Int)

Returns the gate name for the next single qubit gate to be applied to the qubit
at (i, j).
"""
function random_gate!(rqc::RQC, i::Int, j::Int, rng::MersenneTwister)
    gate = rqc.next_gate[i, j]
    if gate > 0
        rqc.next_gate[i, j] = (rqc.next_gate[i, j] + rand(rng, [0,1]))%3 + 1
        rqc.next_gate[i, j] *= -1
        return rqc.single_qubit_gates[gate]
    else
        return ""
    end
end

"""
    function patterns(rqc::RQC)

Generate a dictionary of patterns of qubit pairs to apply two qubit gates to.
The patterns are numbers 1 to 8 like in Boxio_2018 but in a different order.
(Note, the order used in Boxio_2018 is 3, 1, 6, 8, 5, 7, 2, 4)
"""
function patterns(rqc::RQC)
    key = 1
    gp = Dict()
    N = max(rqc.m, rqc.n)
    for v in [1, -1]
        for shift in [0, 1]
            for flip in [false, true]
                gp[key] = [[[i, j], [i, j+v]] for j in 1:2:N
                                              for i in 1+((j+shift)÷2)%2:2:N]
                if flip; gp[key] = map(x->reverse.(x), gp[key]); end
                filter!(gp[key]) do p
                    out_of_bounds(rqc, p)
                end
                key += 1
            end
        end
    end
    gp
end

"""
    function out_of_bounds(rqc::RQC, targets::Array{Array{Int, 1}, 1})

Check if any of the qubits, in the given qubit pair, are out of bounds.
"""
function out_of_bounds(rqc::RQC, targets::Array{Array{Int, 1}, 1})
    ((i, j), (u, v)) = targets
    (0 < i <= rqc.n) && (0 < u <= rqc.n) &&
    (0 < j <= rqc.m) && (0 < v <= rqc.m)
end

"""
    function create_RQC(rows::Int, cols::Int)

Generate a RQC circuit acting on a grid of qubits whose row and column
numbers are (rows, cols).

CZ gates are used by default as the two qubit gates. Setting 'use_iswap' to
true will use iSWAP gates inplace of CZ gates.

Setting 'final_Hadamard_layer' to true will include a layer of Hadamard gates
at the end of the circuit.
"""
function create_RQC(rows::Int, cols::Int, depth::Int,
                    seed::Union{Int, Nothing}=nothing;
                    use_iswap::Bool=false,
                    final_Hadamard_layer::Bool=false)

    if seed === nothing
        rng = MersenneTwister()
    else
        rng = MersenneTwister(seed)
    end

    # Create an empty rqc.
    rqc = RQC(rows, cols)

    # A dictionary to map gate names to qiskit circuit methods that add the
    # corresponding gate to the rqc.
    # TODO: x and y should be replaced by sqrt of x and y
    gates = Dict("h" => rqc.circ.h, "cz" => rqc.circ.cz, "t" => rqc.circ.t,
                 "x" => target -> rqc.circ.rx(pi/2, target),
                 "y" => target -> rqc.circ.ry(pi/2, target),
                 "iswap" => rqc.circ.iswap)

    # A function to add a gate to the rqc with specific target qubits.
    function add_gate!(rqc::RQC, gate::String, target_qubits)
        if !(length(gate) == 0)
            target_qubits = [i + (j-1)*rqc.n - 1 for (i,j) in target_qubits]
            gates[gate](target_qubits...)
        end
    end

    # The rqc starts with a layer of hadamard gates applied to all qubits.
    for i in 1:rqc.n, j in 1:rqc.m
        add_gate!(rqc, "h", [[i, j]])
    end

    # Get the patterns to apply two qubit gates in.
    gate_patterns = patterns(rqc)

    # Loop through the patterns applying the two qubit gates and apply a random
    # single qubit gate to the appropriate qubits.
    pattern = [3, 1, 6, 8, 5, 7, 2, 4]
    two_qubit_gate = use_iswap ? "iswap" : "cz"
    for i in 0:depth-1
        i = pattern[i%8 + 1]

        if !(length(gate_patterns[i]) == 0)
            # Apply the two qubit gates.
            for targets in gate_patterns[i]
                add_gate!(rqc, two_qubit_gate, targets)
            end

            # Apply the single qubit gates
            qubits_hit = reduce(vcat, gate_patterns[i])
            for i in 1:rqc.n, j in 1:rqc.m
                if !([i, j] in qubits_hit)
                    gate = random_gate!(rqc, i, j, rng)
                    add_gate!(rqc, gate, [[i, j]])
                end
            end

            # Update rqc with which qubits were hit by a two qubit gate
            # and so may be hit by a single qubit gate next round.
            for ((i, j), (u, v)) in gate_patterns[i]
                rqc.next_gate[i, j] = abs(rqc.next_gate[i, j])
                rqc.next_gate[u, v] = abs(rqc.next_gate[u, v])
            end
        end

        rqc.circ.barrier()
    end

    if final_Hadamard_layer
        for i in 1:rqc.n, j in 1:rqc.m
            add_gate!(rqc, "h", [[i, j]])
        end
    end

    rqc.circ
end
