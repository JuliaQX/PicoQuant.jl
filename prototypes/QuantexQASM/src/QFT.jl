module QFT

export gen_qft, gen_iqft
include("GateOps.jl")

"""
    gen_qft(q_reg::String, qubit_indices::Vector{Int})

Generates an OpenQASM representation of the quantum Fourier transform
for the given number of qubits.

"""
function gen_qft(q_reg::String, qubit_indices::Vector{Int})
    circuit = ""
    angle(k) = 2*pi / 2^k;
    for (iidx,ival) = Iterators.reverse(enumerate(qubit_indices))
        circuit *= GateOps.apply_gate_h(q_reg, qubit_indices[iidx])
        for (jidx,jval) = Iterators.reverse(enumerate(qubit_indices[1:iidx-1]))
            theta = angle(iidx - jidx);
            circuit *= GateOps.apply_gate_cphase(q_reg, theta, jval, ival)
        end
    end
    for idx = 1:div(length(qubit_indices),2)
        circuit *= GateOps.apply_swap(q_reg, qubit_indices[idx], qubit_indices[length(qubit_indices)-idx+1])
    end
    return circuit
end

"""
    gen_iqft(q_reg::String, qubit_indices::Vector{Int})

Generates an OpenQASM representation of the inverse quantum Fourier transform
for the given number of qubits.

"""
function gen_iqft(q_reg::String, qubit_indices::Vector{Int})
    circuit = ""
    angle(k) = -2*pi / 2^k;
    for (iidx,ival) = enumerate(qubit_indices)
        for (jidx,jval) = enumerate(qubit_indices[1:iidx-1])
            theta = angle(iidx - jidx);
            circuit *= GateOps.apply_gate_cphase(q_reg, theta, jval, ival)
        end
        circuit *= GateOps.apply_gate_h(q_reg, qubit_indices[iidx])

    end
    for idx = 1:div(length(qubit_indices),2)
        circuit *= GateOps.apply_swap(q_reg, qubit_indices[idx], qubit_indices[length(qubit_indices)-idx+1])
    end
    return circuit
end

end
