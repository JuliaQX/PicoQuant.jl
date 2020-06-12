import Base.push!

export AbstractBackend, DSLBackend, InteractiveBackend, save_tensor_data
export backend
export push!

backend = nothing

"Abstract type defining backends which determine the behavior of functions that
act on TensorNetworkCircuits"
abstract type AbstractBackend end

# Function methods to warn the user when a backend isn't initialised.

function save_tensor_data(nothing,
                          node_label::Symbol,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})
    error("Please initialise a backend")
end

function contract_tensors(nothing,
                          A_label::Symbol, A_ncon_indices,
                          B_label::Symbol, B_ncon_indices,
                          C_label::Symbol)
    error("Please initialise a backend")
end

function save_output(nothing, node::Symbol, name::String)
    error("Please initialise a backend")
end

function reshape_tensor(nothing,
                        tensor::Symbol,
                        shape::Union{Array{<:Integer, 1}, Integer})
    error("Please initialise a backend")
end

# *************************************************************************** #
#                               DSL Backend
# *************************************************************************** #

"
The dsl backend will get functions that act on a tensor network circuit to
write dsl commands. It also holds filenames for files containing dsl commands,
tensor data and the result of fully contracting a tensor network circuit.
"
mutable struct DSLBackend <: AbstractBackend
    dsl_filename::String
    tensor_data_filename::String
    output_data_filename::String

    function DSLBackend(dsl::String="contract_network.tl",
                       tensor_data="tensor_data.h5"::String,
                       output::String="")
        # Create an empty dsl file.
        close(open(dsl, "w"))

        # If no output filename given by the user, use the tensor data file
        # to store output.
        if output==""
            output = tensor_data
        end

        global backend
        backend = new(dsl, tensor_data, output)
    end
end

"""
    function push!(dsl::DSLBackend, instruction::String)

Function to append a dsl command to the dsl script contained in a dsl backend.
"""
function push!(dsl::DSLBackend, instruction::String)
    open(dsl.dsl_filename, "a") do io
        write(io, instruction * "\n")
    end
end

"""
    function save_tensor_data(backend::DSLBackend,
                              node_label::Symbol,
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number})

Function to save tensor data to the hdf5 file specified by a dsl backend. It
also appends a load command to the dsl script contained in the backend to load
the saved tensor data as a tensor with the given node label.
"""
function save_tensor_data(backend::DSLBackend,
                          node_label::Symbol,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})

    # Write the tensor data to the file specified by the dsl backend
    # and append commands to the dsl script to the tensor.
    h5open(backend.tensor_data_filename, "cw") do file
        tensor_label = string(tensor_label)
        write(file, tensor_label, tensor_data)
        push!(backend, "tensor $node_label $tensor_label")
    end
end

"""
    function contract_tensors(backend::DSLBackend,
                              A_label::Symbol, A_ncon_indices::Array{<:Integer, 1},
                              B_label::Symbol, B_ncon_indices::Array{<:Integer, 1},
                              C_label::Symbol)

Function to add dsl commands to a dsl backend's script that contract two tensors
A and B to create a third tensor C.
"""
function contract_tensors(backend::DSLBackend,
                          A_label::Symbol, A_ncon_indices::Array{<:Integer, 1},
                          B_label::Symbol, B_ncon_indices::Array{<:Integer, 1},
                          C_label::Symbol)

    # Create and save the dsl commands for contracting A and B.
    ncon_str = "ncon " * string(C_label) * " "
    ncon_str *= string(A_label) * " " * join(A_ncon_indices, ",") * " "
    ncon_str *= string(B_label) * " " * join(B_ncon_indices, ",")
    push!(backend, ncon_str)

    # Delete the old tensors.
    push!(backend, "del " * string(A_label))
    push!(backend, "del " * string(B_label))
end

"""
    function save_output(backend::DSLBackend, node::Symbol, name::String)

Function to save the result of fully contracting a network under the given
name.
"""
function save_output(backend::DSLBackend, node::Symbol, name::String="result")
    command = "save $(node) $(backend.output_data_filename) $name"
    push!(backend, command)
end

"""
    function reshape_tensor(backend::DSLBackend, tensor::Symbol, shape)

Function to add dsl command that reshapes a given tensor.
"""
function reshape_tensor(backend::DSLBackend,
                        tensor::Symbol,
                        shape::Union{Array{<:Integer, 1}, Integer})
    command = "reshape $tensor " * join(shape, ",")
    push!(backend, command)
end

# *************************************************************************** #
#                             Interactive Backend
# *************************************************************************** #


"The Interactive backend gets functions to act on a TensorNetworkCircuit
interactively."
mutable struct InteractiveBackend <: AbstractBackend
    use_gpu::Bool
    memory_size_mb::Int64
    tensors::Dict{Symbol, Array{<:Number}}

    function InteractiveBackend(use_gpu::Bool=false, memory_size_mb::Int64=0)
        global backend
        tensors = Dict{Symbol, Array{<:Number}}()
        backend = new(use_gpu, memory_size_mb, tensors)
    end
end

"""
    function save_tensor_data(backend::InteractiveBackend,
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number})

Function to save tensor data to a dictionary in an interactive backend.
"""
function save_tensor_data(backend::InteractiveBackend,
                          node_label::Symbol,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})
    # TODO: node_label isn't used here but could be used when a tensor registry
    # is added to avoid duplication of tensor data.
    backend.tensors[tensor_label] = tensor_data
end

"""
    function contract_tensors(backend::InteractiveBackend,
                              A_label::Symbol, A_ncon_indices::Array{<:Integer, 1},
                              B_label::Symbol, B_ncon_indices::Array{<:Integer, 1},
                              C_label::Symbol)

Function to interactively contract two tensors A and B to create a tensor C.
"""
function contract_tensors(backend::InteractiveBackend,
                          A_label::Symbol, A_ncon_indices::Array{<:Integer, 1},
                          B_label::Symbol, B_ncon_indices::Array{<:Integer, 1},
                          C_label::Symbol)

    # Get the tensor data for A and B and compute the tensor data for C.
    A_data = backend.tensors[A_label]
    B_data = backend.tensors[B_label]
    C_data = contract_tensors((A_data, B_data),
                              (A_ncon_indices, B_ncon_indices))
    if typeof(C_data) <: Number
        C_data = [C_data]
    end

    # Save the new tensor and delete the tensors that were contracted.
    backend.tensors[C_label] = C_data
    delete!(backend.tensors, A_label)
    delete!(backend.tensors, B_label)
end

"""
    function save_output(backend::InteractiveBackend, node::Symbol, name::String)

Function to save the result of fully contracting a network under the given
name.
"""
function save_output(backend::InteractiveBackend,
                     node::Symbol, name::String="result")
    name = Symbol(name)
    backend.tensors[name] = backend.tensors[node]
end

"""
    function reshape_tensor(backend::InteractiveBackend, tensor::Symbol, shape)

Function to reshape a given tensor.
"""
function reshape_tensor(backend::InteractiveBackend,
                        tensor::Symbol,
                        shape::Union{Array{<:Integer, 1}, Integer})
    backend.tensors[tensor] = reshape_tensor(backend.tensors[tensor], shape)
end
