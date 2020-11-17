"Abstract type defining backends which determine the behavior of functions that
act on TensorNetworkCircuits"
abstract type AbstractBackend end

"""
A struct to record various costs of storing and contracting a tensor network.
"""
mutable struct Metrics
    # Maximum number of elements in a single tensor
    max_tensor_size::Int

    # Total number of elements required to store the tensor network and
    # intermidiate tensors during contraction.
    total_space_allocated::Int

    # Number of multiplications required to contract a network
    flops::Int

    Metrics() = new(0, 0, 0)
end

"""
    function record_compute_costs!(backend::AbstractBackend,
                                   open_dims::Array{Int, 1},
                                   contracted_dims::Array{Int, 1})

Function to calculate the space and time costs of contracting two tensors
and add them to the metrics contained in the given backend.
"""
function record_compute_costs!(backend::AbstractBackend,
                               open_dims::Array{Int, 1},
                               contracted_dims::Array{Int, 1})

    space_cost = prod(open_dims)
    record_memory_costs!(backend, space_cost)

    flops_cost = space_cost * prod(contracted_dims)
    backend.metrics.flops += flops_cost

    flops_cost, space_cost
end

"""
    function record_memory_costs!(backend::AbstractBackend,
                                  tensor_memory::Int)

Function to add the space cost of a tensor to the metrics contained in
the given backend.
"""
function record_memory_costs!(backend::AbstractBackend,
                              tensor_memory::Int)

    backend.metrics.total_space_allocated += tensor_memory
    if tensor_memory > backend.metrics.max_tensor_size
       backend.metrics.max_tensor_size = tensor_memory
    end
end

include("backends/interactive.jl")
include("backends/dsl.jl")

export AbstractBackend
export save_tensor_data, load_tensor_data, save_output
export contract_tensors, record_costs!
export reshape_tensor, permute_tensor
export decompose_tensor!, view_tensor!, delete_tensor!
