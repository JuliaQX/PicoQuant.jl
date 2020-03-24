module VQE

#export min_expec_val, expec_val, ansatz_gen, run_circuit_with_qiskit

include("GateOps.jl")
include("Utils.jl")

using PyCall
using Optim
using UUIDs

###############################################################################
#               Module setup
###############################################################################

#Track the number of circuits generated
cct_runs = 0

#Create temporary directory hold intermdiate QASM files
cct_dir = "/tmp/cct_dir_" * string(UUIDs.uuid4()) * "/" 

function __init__()
    init_env()
    mkdir(cct_dir)
end

"""
    init_env()

Installs necessary packages to run the VQE method. Qiskit is installed into the 
local Python environment packaged weith Julia's conda.
"""
function init_env()
    run(`$(PyCall.python) -m pip install qiskit`)
end

"""
Cleanup the generated files.
"""
function refresh_dir()
    rm(cct_dir, recursive=true)
    mkdir(cct_dir)
end

###############################################################################
#               Circuit evaulation functions
###############################################################################

"""
    run_circuit_with_qiskit(cct::String, num_samples::Int64)

Generate an ansatz using qubit register `q_reg` on indices `qubit_indices` with angle `theta`.
Sample ansatz uses R_x(theta) gate.
"""
function run_circuit_with_qiskit(cct::String, num_samples::Int64)
    global cct_runs += 1
    qiskit = PyCall.pyimport("qiskit");
    qiskit_execute = PyCall.pyimport("qiskit.execute");

    qubit_label_q = "qr"
    qubit_label_c = "cr"
    q_count = 1
    c_count = q_count

    qubit_indices = collect(0:q_count-1)

    qasm_file = GateOps.gen_qasm_header(q_count, c_count, qubit_label_q, qubit_label_c )
    qasm_file *= cct

    cct_file = open(cct_dir * "cct_$cct_runs.qasm", "w") 
    for i in Utils.format_string_nl(qasm_file)
        write(cct_file, i)
    end
    close(cct_file) 

    circ = qiskit.QuantumCircuit.from_qasm_file(cct_dir * "cct_$cct_runs.qasm")

    #Run the circuit
    simulator = qiskit.Aer.get_backend("qasm_simulator")
    job = qiskit_execute.execute(circ, simulator, shots=num_samples)
    result = job.result()
    counts = result.get_counts(circ)
    print("\nTotal count for 00 and 11 are:", counts)

    return counts;
end

"""
    ansatz_gen(q_reg::String, qubit_indices::Vector{Int}, theta::Float64)

Generate an ansatz using qubit register `q_reg` on indices `qubit_indices` with angle `theta`.
Sample ansatz uses R_x(theta) gate.
"""
function ansatz_gen(q_reg::String, qubit_indices::Vector{Int}, theta::Float64)
    ansatz = GateOps.reset_reg(qubit_indices, q_reg)
    for idx in qubit_indices
        ansatz *= GateOps.apply_gate_rx(q_reg, theta, idx)
    end
    return ansatz
end

"""
    expec_val(theta::Real)

Evaluate the expectation value of the given test-case Pauli-Z.
"""
function expec_val(theta::Real, num_samples::Int64)
    q_reg="qr"
    c_reg="cr"
    qubit_indices=[0]
    circ = ""
    circ *= ansatz_gen(q_reg, qubit_indices, theta)
    circ *= GateOps.measure_qubits_all(qubit_indices, qubit_indices, q_reg, c_reg)

    c = run_circuit_with_qiskit(circ, num_samples, )

    p0 = [val for (key, val) in c if key=="0"];
    p1 = [val for (key, val) in c if key=="1"];

    if length(p0) == 0
        p0 = [0]
    elseif length(p1) == 0
        p1 = [0]
    end
    return (p0[1] - p1[1]) / num_samples
end

"""
    min_expec_val(theta::Real)

Find the minimum expectation value of the given Hamiltonian (Pauli-Z).
Uses Optim.jl package to perform classical optimisation.
"""
function min_expec_val(start_val::Real, num_samples::Int64=1000)
    #Clear previous data before performing new experiment
    global cct_runs = 0
    refresh_dir()

    f(x) = expec_val(x[1], num_samples)
    min_val = Optim.optimize(f, [start_val], Optim.BFGS())
    return min_val
end

end