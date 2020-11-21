using TensorOperations
using EllipsisNotation
using DataStructures
import Base: size, length, getindex

export MPSState, size, length, getindex

"""Data structure to represent an MPS state"""
struct MPSState{T, N} <: AbstractArray{T, N}
    # stores the data from each mps node
    data_tensors::Array{Array{T}, 2}
    # used to store the meta data for each mps node with index locations
    nodes::Array{Node, 1}
    # the order mapping of output qubits to classical measurement bits
    ordering::Array{Int, 1}
end

"""
    function MPSState(network::TensorNetworkCircuit, mps_nodes::Array{Symbol, 1})

Constructor for MPS state which takes an tensor network circuit and list of mps
nodes and populates fields which enables one ot efficiently retrieve amplitudes
via the array interface
"""
function MPSState(::Type{T}, network::TensorNetworkCircuit, mps_nodes::Array{Symbol, 1}) where {T <: Number}
    n = length(mps_nodes)
    # we create arrays to store node data structure which contains the meta data
    # and data arrays which we create for each mps tensor
    nodes = Array{Node, 1}(undef, n)
    data_tensors = Array{Array{T}, 2}(undef, (n, 2))
    output_positions = Array{Int64, 1}(undef, length(mps_nodes))
    for i = 1:length(mps_nodes)
        node_label = mps_nodes[i]
        # output_index = network.output_qubits[i]
        output_positions[i] = findfirst(x -> x in network.output_qubits,
                                    network.nodes[node_label].indices)
        other_positions = collect(
                    setdiff(
                        OrderedSet(1:length(network.nodes[node_label].indices)),
                        output_positions[i])
                    )
        tensor = permutedims(
                    load_tensor_data(network, node_label),
                                     (output_positions[i], other_positions...))
        nodes[i] = Node(network.nodes[node_label].indices[[other_positions...]],
                        collect(size(tensor)),
                        Symbol("n_$i"))
        data_tensors[i, :] = [tensor[x,..] for x in 1:2]
    end
    ordering = Array{Int, 1}(undef, n)
    for (i, e) in zip(output_positions, network.qubit_ordering)
        ordering[e] = i
    end
    MPSState{T, n}(data_tensors, nodes, ordering)
end

function MPSState(network::TensorNetworkCircuit, mps_nodes::Array{Symbol, 1})
    MPSState(ComplexF32, network::TensorNetworkCircuit, mps_nodes::Array{Symbol, 1})
end

"""
    function size(a::MPSState{T, N}) where {T, N}

Return the dimensions in each dimension
TODO: generalise to N != 2
"""
function size(a::MPSState{T, N}) where {T, N}
    Tuple([2 for _ in 1:N])
end

"""
    function length(a::MPSState{T, N}) where {T, N}

Return the total size of the all amplitudes
TODO: generalise to N != 2
"""
function length(a::MPSState{T, N}) where {T, N}
    2^N
end

"""
    function getindex(a::MPSState{T, N}, i::Vararg{Int, N}) where {T, N}

Override the getindex function for abstract array which allows to use the
array notation [i_1, i_2,..., i_N] to access amplitudes
"""
function getindex(a::MPSState{T, N}, i::Vararg{Int, N}) where {T, N}
    n1 = a.nodes[1]
    conf_map = x -> (x % 2 == 1) ? 1 : 2
    n1_data = a.data_tensors[1, conf_map(i[a.ordering[1]])]
    for idx in 2:length(a.nodes)
        n2 = a.nodes[idx]
        common, remaining = sort_indices(n1, n2)
        indices = create_ncon_indices(n1, n2, common, remaining)
        n1 = Node(remaining, Int64[], :n1)
        n2_data = a.data_tensors[idx, conf_map(i[a.ordering[idx]])]
        n1_data = ncon([n1_data, n2_data],  indices)
    end
    scalar(n1_data)
end

"""
    function getindex(a::MPSState, conf::String)

Override the getindex method to accept strings to enable retrieval of amplitudes
from strings
"""
function getindex(a::MPSState, conf::String)
    a[[x == '0' ? 1 : 2 for x in conf]...]
end
