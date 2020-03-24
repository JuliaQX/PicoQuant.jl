module GateOps

export GateOps
include("QASM_Map.jl")

export bloch_angles, generate_gate, pi_convert
export apply_gate_u, apply_gate_x, apply_gate_y, apply_gate_z, apply_gate_h
export apply_gate_cx, apply_gate_cy, apply_gate_cz, apply_gate_cphase, apply_gate_ccx
export apply_gate_rx, apply_gate_ry, apply_gate_rz, apply_gate_s, apply_gate_t, apply_swap
 
#Define map of OpenQASM structures for use in generation of appropriate structures
qasm_map = define_qasm_mapping()

struct bloch_angles
    θ::Real
    ϕ::Real
    λ::Real
end

"""
    generate_gate(gate_name)

Generates an OpenQASM-style gate structure string.
The internals of {} can be replace with the appropriate
values.

# Examples
```julia-repl
julia> generate_gate("Q")
\"gate Q {}\"
```
"""
function generate_gate(gate_name::String)
   return "gate " * gate_name * " {}" 
end

"""
    pi_convert(value::Real)

Converts the Julia internal Irrational pi string to the string pi, or the value.
Prevents pi string being replaced by π = 3.1415926535897..., as pi is defined
in OpenQASM.

# Examples
```julia-repl
julia> pi
\"π = 3.1415926535897...\"
```
```julia-repl
julia> pi_convert(pi)
\"pi\"
```
```julia-repl
julia> pi_convert(pi/2)
\"1.5707963267948966\"
```
"""
function pi_convert(value::Real)
    return typeof(value)<:Irrational{:π} ? "pi" : value
end

"""
    apply_gate_u(angles::bloch_angles, q_reg::String, q_idx::Union{Int, Nothing}=nothing)

Generates the OpenQASM syntax for a U(theta,phi,lambda) unitary as per the OpenQASM specification
"""
function apply_gate_u(angles::bloch_angles, q_reg::String, q_idx::Union{Int, Nothing}=nothing)
   
    result = replace(qasm_map["U"], "theta"=>pi_convert(angles.θ))
    result = replace(result, "phi"=>pi_convert(angles.ϕ))
    result = replace(result, "lambda"=>pi_convert(angles.λ))
    result = replace(result, "q_reg"=>q_reg)
    
    #If no index given, apply same ops across entire register
    if q_idx == nothing 
        result = replace(result, "[q_idx]"=>"")
    else # Otherwise apply op to given qubit index
        result = replace(result, "q_idx"=>q_idx)
    end
    return result
end

"""
    apply_gate_x(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply Pauli-X to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_x(q_reg::String, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(pi, 0, pi), q_reg, q_idx)
end
"""
    apply_gate_y(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply Pauli-Y to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_y(q_reg::String, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(pi, pi/2, pi/2), q_reg, q_idx)
end
"""
    apply_gate_z(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply Pauli-Z to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_z(q_reg::String, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(0, 0, pi), q_reg, q_idx)
end
"""
    apply_gate_h(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply Hadamard to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_h(q_reg::String, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(pi/2, 0, pi), q_reg, q_idx)
end

"""
    apply_gate_t(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply T gate to qubit `q_idx` in register `q_reg`. If `is_adjoint==true`, applies T^{dagger}
"""
function apply_gate_t(q_reg::String, q_idx::Union{Int, Nothing}=nothing, is_adjoint::Union{Bool, Nothing}=nothing)
    #If no index given, apply same ops across entire register
    if is_adjoint == nothing
        result = qasm_map["T"]
    else
        result = qasm_map["TDG"]
    end        
    
    result = replace(result, "q_reg"=>q_reg)
    
    if q_idx == nothing
        result = replace(result, "[q_idx]"=>"")
    else # Otherwise apply op to given qubit index
        result = replace(result, "q_idx"=>q_idx)
    end
    return result
end

"""
    apply_gate_t(q_reg::String, q_idx::Union{Int, Nothing}=nothing)

    Apply S gate to qubit `q_idx` in register `q_reg`. If `is_adjoint==true`, applies S^{dagger}
"""
function apply_gate_s(q_reg::String, q_idx::Union{Int, Nothing}=nothing, is_adjoint::Union{Bool, Nothing}=nothing)
    #If no index given, apply same ops across entire register
    if is_adjoint == nothing
        result = qasm_map["S"]
    else
        result = qasm_map["SDG"]
    end        
    
    result = replace(result, "q_reg"=>q_reg)
    
    if q_idx == nothing
        result = replace(result, "[q_idx]"=>"")
    else # Otherwise apply op to given qubit index
        result = replace(result, "q_idx"=>q_idx)
    end
    return result
end

"""
    apply_gate_rx(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply R_x(theta) gate to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_rx(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(θ,-pi/2,pi/2), q_reg, q_idx)
end

"""
    apply_gate_ry(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply R_y(theta) gate to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_ry(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(θ,0,0), q_reg, q_idx)
end

"""
    apply_gate_rz(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply R_z(theta) gate to qubit `q_idx` in register `q_reg`.
"""
function apply_gate_rz(q_reg::String, ϕ::Real, q_idx::Union{Int, Nothing}=nothing)
    return apply_gate_u( bloch_angles(0, 0, ϕ), q_reg, q_idx)
end

"""
    apply_gate_cx(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply controlled-not (CX) gate.
"""
function apply_gate_cx(q_reg::String, q_ctrl_idx::Int, q_tgt_idx::Int)
    result = replace(qasm_map["CX"], "c"=>"$q_reg[$q_ctrl_idx]")
    result = replace(result, "t"=>"$q_reg[$q_tgt_idx]")
    return result
end

"""
    apply_gate_cx(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply controlled-not (CY) gate.
"""
function apply_gate_cy(q_reg::String, q_ctrl_idx::Int, q_tgt_idx::Int)
    result =  apply_gate_u( bloch_angles(0, 0, -pi/2), q_reg, q_tgt_idx);
    result *= apply_gate_cx(q_reg, q_ctrl_idx, q_tgt_idx)
    result *= apply_gate_u( bloch_angles(0, 0, pi/2), q_reg, q_tgt_idx);
    return result
end

"""
    apply_gate_cx(q_reg::String, θ::Real, q_idx::Union{Int, Nothing}=nothing)

    Apply controlled-not (CZ) gate.
"""
function apply_gate_cz(q_reg::String, q_ctrl_idx::Int, q_tgt_idx::Int)
    result =  apply_gate_h(q_reg, q_tgt_idx)
    result *= apply_gate_cx(q_reg, q_ctrl_idx, q_tgt_idx)
    result *= apply_gate_h(q_reg, q_tgt_idx)
    return result 
end

"""
    apply_gate_ccx(q_reg::String, q_ctrl_idx1::Int, q_ctrl_idx2::Int, q_tgt_idx::Int)

    Apply controlled-controlled-not (CCX, Tofolli) gate.
"""
function apply_gate_ccx(q_reg::String, q_ctrl_idx1::Int, q_ctrl_idx2::Int, q_tgt_idx::Int)
    result =  apply_gate_h(q_reg, q_tgt_idx)
    result *= apply_gate_cx(q_reg, q_ctrl_idx2, q_tgt_idx)
    result *= apply_gate_t(q_reg, q_tgt_idx, true)
    result *= apply_gate_cx(q_reg, q_ctrl_idx1, q_tgt_idx)
    result *= apply_gate_t(q_reg, q_tgt_idx, false)
    result *= apply_gate_cx(q_reg, q_ctrl_idx2, q_tgt_idx)
    result *= apply_gate_t(q_reg, q_tgt_idx, true)
    result *= apply_gate_cx(q_reg, q_ctrl_idx1, q_tgt_idx)
    result *= apply_gate_t(q_reg, q_ctrl_idx2, false)
    result *= apply_gate_t(q_reg, q_tgt_idx, false)
    result *= apply_gate_h(q_reg, q_tgt_idx)
    result *= apply_gate_cx(q_reg, q_ctrl_idx1, q_ctrl_idx2)
    result *= apply_gate_t(q_reg, q_ctrl_idx1, false)
    result *= apply_gate_t(q_reg, q_ctrl_idx2, true)
    result *= apply_gate_cx(q_reg, q_ctrl_idx1, q_ctrl_idx2)
    return result 
end

"""
    apply_gate_cphase(q_reg::String, λ::Real, q_ctrl_idx::Int, q_tgt_idx::Int)

    Apply controlled-phase (Controlled-[[1,0],[0, exp(iλ)]]) gate.
"""
function apply_gate_cphase(q_reg::String, λ::Real, q_ctrl_idx::Int, q_tgt_idx::Int)
    # [[1,0],[0, exp(iλ)]]
    result =  apply_gate_u( bloch_angles(0, 0, λ/2), q_reg, q_ctrl_idx);
    result *= apply_gate_cx(q_reg, q_ctrl_idx, q_tgt_idx);
    result *= apply_gate_u( bloch_angles(0, 0, -λ/2), q_reg, q_tgt_idx);
    result *= apply_gate_cx(q_reg, q_ctrl_idx, q_tgt_idx);
    result *= apply_gate_u( bloch_angles(0, 0, λ/2), q_reg, q_tgt_idx);
end

"""
    apply_swap(q_reg::String, q1::Int, q2::Int)

    Swap qubit states between q1 and q2
"""
function apply_swap(q_reg::String, q1::Int, q2::Int)
    result =  apply_gate_cx(q_reg, q1, q2)
    result *= apply_gate_cx(q_reg, q2, q1)
    result *= apply_gate_cx(q_reg, q1, q2)
    return result
end

function gen_qasm_header(num_q::Int64, num_c::Int64, label_q::String, label_c::String)
    h = "OPENQASM 2.0;include \"qelib1.inc\";"
    h *= "qreg $label_q[$num_q];";
    h *= "creg $label_c[$num_c];";
    return h
end

function measure_qubit(idx_q::Int64, idx_c::Int64, label_q::String, label_c::String)
    return "measure $label_q[$idx_q] -> $label_c[$idx_c];"
end

function measure_qubits_all(idx_q::Vector{Int64}, idx_c::Vector{Int64}, label_q::String, label_c::String)
    m = ""
    for i in idx_q
        m*="measure $label_q[$i] -> $label_c[$i];"
    end
    return m;
end

function reset_reg(idx_q::Vector{Int64}, label_q::String)
    result = ""
    for i in idx_q
        result *= "reset $label_q[$i];"
    end
    return result;
end

end
