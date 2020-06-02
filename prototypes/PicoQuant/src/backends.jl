
export AbstractBackend, DSLWriter, InteractiveBackend, save_tensor_data
export backend

backend = nothing

# *************************************************************************** #
#                             Backend types
# *************************************************************************** #

"Abstract type defining executers which determine the behavior of contract
functions"
abstract type AbstractBackend end

"The dsl writer will get the contract functions to write dsl commands."
mutable struct DSLWriter <: AbstractBackend
    dsl_filename::String
    tensor_data_filename::String
    output_data_filename::String

    function DSLWriter(dsl::String="contract_network.tl",
                       tensor_data="tensor_data.h5"::String,
                       output::String="")
        # create empty dsl file
        close(open(dsl, "w"))
        global backend
        backend = new(dsl, tensor_data, output)
    end
end

function push!(dsl::DSLWriter, instruction::String)
    open(dsl.dsl_filename, "a") do io
        write(io, instruction * "\n")
    end
end

"Interactive executer which gets the contract functions to act on tensor data
in a TensorNetworkCircuit interactively."
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

# *************************************************************************** #
#                  Save tensor data according to backend
# *************************************************************************** #

function save_tensor_data(nothing, tensor_label, tensor_data)
    error("Please create a backend before creating a network")
end

"""
    function save_tensor_data(backend::DSLWriter,
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number})

Function to save tensors data to the hdf5 file specified by a dsl writer.
"""
function save_tensor_data(backend::DSLWriter,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})

    # Write the tensor data to a the file specified by the executer.
    h5open(backend.tensor_data_filename, "cw") do file
        tensor_label = string(tensor_label)
        write(file, tensor_label, tensor_data)
    end
end

"""
    function save_tensor_data(backend::InteractiveBackend,
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number})

Function to save tensors data to a dictionary in an interactive backend.
"""
function save_tensor_data(backend::InteractiveBackend,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})

    # Save the tensor data to a the interactive executer.
    backend.tensors[tensor_label] = tensor_data
end
