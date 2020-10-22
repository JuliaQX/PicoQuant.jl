export InteractiveBackend


# *************************************************************************** #
#                             Interactive Backend
# *************************************************************************** #

"The Interactive backend gets functions to act on a TensorNetworkCircuit
interactively."
mutable struct InteractiveBackend{T} <: AbstractBackend
    tensors::Dict{Symbol, T}
    metrics::Metrics

    function InteractiveBackend{T}() where {T}
        tensors = Dict{Symbol, T}()
        new(tensors, Metrics())
    end
    # Default construct with ComplexF32
    function InteractiveBackend()
        tensors = Dict{Symbol, Array{ComplexF32}}()
        return InteractiveBackend{Array{ComplexF32}}()
    end
end

"""
    save_tensor_data(backend::InteractiveBackend{T},
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number}) where T <: AbstractArray

Function to save tensor data to a dictionary in an interactive backend.
"""
function save_tensor_data(backend::InteractiveBackend{T},
                          tensor_label::Symbol,
                          tensor_data::U) where {T <: AbstractArray, U <: AbstractArray}
    backend.tensors[tensor_label] = convert(T, tensor_data)
end

"""
    load_tensor_data(backend::InteractiveBackend{T},
                              tensor_label::Symbol) where T <: AbstractArray

Function to load tensor data from backend storage (if present)
"""
function load_tensor_data(backend::InteractiveBackend{T},
                          tensor_label::Symbol) where T <: AbstractArray
    if tensor_label in keys(backend.tensors)
        return backend.tensors[tensor_label]
    end
end


"""
    contract_tensors(backend::InteractiveBackend{T},
                              A_label::Symbol, A_ncon_indices::Array{Int, 1},
                              B_label::Symbol, B_ncon_indices::Array{Int, 1},
                              C_label::Symbol) where T <: AbstractArray

Function to interactively contract two tensors A and B to create a tensor C.
"""
function contract_tensors(backend::InteractiveBackend{T},
                          A_label::Symbol, A_ncon_indices::Array{Int, 1},
                          B_label::Symbol, B_ncon_indices::Array{Int, 1},
                          C_label::Symbol) where T <: AbstractArray

    # Get the tensor data for A and B and compute the tensor data for C.
    A_data = backend.tensors[A_label]
    B_data = backend.tensors[B_label]
    C_data = contract_tensors((A_data, B_data),
                              (A_ncon_indices, B_ncon_indices))

    # Save the new tensor and delete the tensors that were contracted.
    save_tensor_data(backend, C_label, C_data)
    delete_tensor!(backend, A_label)
    delete_tensor!(backend, B_label)
end

"""
    save_output(backend::InteractiveBackend{T},
                         node::Symbol, name::String) where T <: AbstractArray

Function to save the result of fully contracting a network under the given
name.
"""
function save_output(backend::InteractiveBackend{T},
                     node::Symbol, name::String="result") where T <: AbstractArray
    name = Symbol(name)
    save_tensor_data(backend, name, backend.tensors[node])
end

"""
    reshape_tensor(backend::InteractiveBackend{T},
                            tensor::Symbol,
                            groups::Array{Array{Int, 1}, 1}) where T <: AbstractArray

Function to reshape a given tensor.
"""
function reshape_tensor(backend::InteractiveBackend{T},
                        tensor::Symbol,
                        groups::Array{Array{Int, 1}, 1}) where T <: AbstractArray
    tensor_dims = size(backend.tensors[tensor])
    backend.tensors[tensor] = reshape_tensor(backend.tensors[tensor], [prod(tensor_dims[x]) for x in groups])
end


"""
    permute_tensor(backend::InteractiveBackend{T},
                            tensor::Symbol, axes::Array{Int, 1}) where T <: AbstractArray

Function to permute the axes of the given tensor
"""
function permute_tensor(backend::InteractiveBackend{T},
                        tensor::Symbol,
                        axes::Array{Int, 1}) where T <: AbstractArray
    backend.tensors[tensor] = permute_tensor(backend.tensors[tensor], axes)
end

"""
    decompose_tensor!(backend::InteractiveBackend{T},
                               tensor::Symbol,
                               left_positions::Array{Int, 1},
                               right_positions::Array{Int, 1};
                               threshold::AbstractFloat=1e-13,
                               max_rank::Int=0,
                               left_label::Symbol,
                               right_label::Symbol) where T <: AbstractArray

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
                           right_label::Symbol) where T <: AbstractArray

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
    delete_tensor!(backend::InteractiveBackend{T}, tensor_label::Symbol) where T <: AbstractArray

Mark tensor for deletion
"""
function delete_tensor!(backend::InteractiveBackend{T}, tensor_label::Symbol) where T <: AbstractArray
    delete!(backend.tensors, tensor_label)
end


"""
    view_tensor!(backend::InteractiveBackend{T}, view_node, node, bond_idx, bond_range) where T <: AbstractArray

Create a view on a tensor
"""
function view_tensor!(backend::InteractiveBackend{T}, view_node, node, bond_idx, bond_range) where T <: AbstractArray
    node_data = backend.tensors[node]
    backend.tensors[view_node] = tensor_view(node_data, bond_idx, bond_range)
end
