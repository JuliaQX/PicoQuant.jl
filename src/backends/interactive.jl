export InteractiveBackend

# *************************************************************************** #
#                             Interactive Backend
# *************************************************************************** #

"The Interactive backend gets functions to act on a TensorNetworkCircuit
interactively."
mutable struct InteractiveBackend{T} <: AbstractBackend where {T <: Number}
    use_gpu::Bool
    memory_size_mb::Int
    tensors::Dict{Symbol, Array{T}}
    metrics::Metrics

    function InteractiveBackend{T}(use_gpu::Bool=false, memory_size_mb::Int=0) where {T <: Number}
        tensors = Dict{Symbol, Array{T}}()
        new(use_gpu, memory_size_mb, tensors, Metrics())
    end
    # Default construct with ComplexF64
    function InteractiveBackend(use_gpu::Bool=false, memory_size_mb::Int=0)
        return InteractiveBackend{ComplexF32}(use_gpu, memory_size_mb)
    end
end

"""
    function save_tensor_data(backend::InteractiveBackend{T},
                              tensor_label::Symbol,
                              tensor_data::Array{U}) where {T <: Number, U <: Number}

Function to save tensor data to a dictionary in an interactive backend.
"""
function save_tensor_data(backend::InteractiveBackend{T},
                          node_label::Symbol,
                          tensor_label::Symbol,
                          tensor_data::Array{U}) where {T <: Number, U <: Number}
    # TODO: node_label isn't used here but could be used when a tensor registry
    # is added to avoid duplication of tensor data.
    backend.tensors[tensor_label] = convert(Array{T}, tensor_data)
end

"""
    function load_tensor_data(backend::InteractiveBackend{T},
                              tensor_label::Symbol) where {T <: Number}

Function to load tensor data from backend storage (if present)
"""
function load_tensor_data(backend::InteractiveBackend{T},
                          tensor_label::Symbol) where {T <: Number}
    if tensor_label in keys(backend.tensors)
        return backend.tensors[tensor_label]
    end
end


"""
    function contract_tensors(backend::InteractiveBackend{T},
                              A_label::Symbol, A_ncon_indices::Array{Int, 1},
                              B_label::Symbol, B_ncon_indices::Array{Int, 1},
                              C_label::Symbol) where {T <: Number}

Function to interactively contract two tensors A and B to create a tensor C.
"""
function contract_tensors(backend::InteractiveBackend{T},
                          A_label::Symbol, A_ncon_indices::Array{Int, 1},
                          B_label::Symbol, B_ncon_indices::Array{Int, 1},
                          C_label::Symbol) where {T <: Number}

    # Get the tensor data for A and B and compute the tensor data for C.
    A_data = backend.tensors[A_label]
    B_data = backend.tensors[B_label]
    C_data = contract_tensors((A_data, B_data),
                              (A_ncon_indices, B_ncon_indices))

    # Save the new tensor and delete the tensors that were contracted.
    save_tensor_data(backend, :fixme, C_label, C_data)
    delete_tensor!(backend, A_label)
    delete_tensor!(backend, B_label)
end

"""
    function save_output(backend::InteractiveBackend{T},
                         node::Symbol, name::String) where {T <: Number}

Function to save the result of fully contracting a network under the given
name.
"""
function save_output(backend::InteractiveBackend{T},
                     node::Symbol, name::String="result") where {T <: Number}
    name = Symbol(name)
    save_tensor_data(backend, :fixme, name, backend.tensors[node])
end

"""
    function reshape_tensor(backend::InteractiveBackend{T},
                            tensor::Symbol,
                            groups::Array{Array{Int, 1}, 1}) where {T <: Number}

Function to reshape a given tensor.
"""
function reshape_tensor(backend::InteractiveBackend{T},
                        tensor::Symbol,
                        groups::Array{Array{Int, 1}, 1}) where {T <: Number}
    tensor_dims = size(backend.tensors[tensor])
    backend.tensors[tensor] = reshape_tensor(backend.tensors[tensor], [prod(tensor_dims[x]) for x in groups])
end


"""
    function permute_tensor(backend::InteractiveBackend{T},
                            tensor::Symbol, axes::Array{Int, 1}) where {T <: Number}

Function to permute the axes of the given tensor
"""
function permute_tensor(backend::InteractiveBackend{T},
                        tensor::Symbol,
                        axes::Array{Int, 1}) where {T <: Number}
    backend.tensors[tensor] = permute_tensor(backend.tensors[tensor], axes)
end

"""
    function decompose_tensor!(backend::InteractiveBackend{T},
                               tensor::Symbol,
                               left_positions::Array{Int, 1},
                               right_positions::Array{Int, 1};
                               threshold::AbstractFloat=1e-13,
                               max_rank::Int=0,
                               left_label::Symbol,
                               right_label::Symbol) where {T <: Number}

Function to decompose a single tensor into two tensors and return the dimension
of the newly created virtual edge.
"""
function decompose_tensor!(backend::InteractiveBackend{T},
                           tensor::Symbol,
                           left_positions::Array{Int, 1},
                           right_positions::Array{Int, 1};
                           threshold::AbstractFloat=1e-13,
                           max_rank::Int=0,
                           left_label::Symbol,
                           right_label::Symbol) where {T <: Number}

    node_data = backend.tensors[tensor]

    (B, C, chi) = decompose_tensor(node_data,
                              left_positions,
                              right_positions,
                              threshold=threshold,
                              max_rank=max_rank)

    backend.tensors[left_label] = B
    backend.tensors[right_label] = C

    delete_tensor!(backend, tensor)
    chi
end

"""
    function delete_tensor!(backend::InteractiveBackend{T}, tensor_label::Symbol) where {T <: Number}

Mark tensor for deletion
"""
function delete_tensor!(backend::InteractiveBackend{T}, tensor_label::Symbol) where {T <: Number}
    delete!(backend.tensors, tensor_label)
end


"""
    function view_tensor!(backend::InteractiveBackend{T}, view_node, node, bond_idx, bond_range) where {T <: Number}

Create a view on a tensor
"""
function view_tensor!(backend::InteractiveBackend{T}, view_node, node, bond_idx, bond_range) where {T <: Number}
    node_data = backend.tensors[node]
    backend.tensors[view_node] = tensor_view(node_data, bond_idx, bond_range)
end
