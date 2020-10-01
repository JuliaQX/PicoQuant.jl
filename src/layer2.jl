import Random.shuffle
using Logging

export random_contraction_plan, inorder_contraction!
export contract_pair!, contract_network!, full_wavefunction_contraction!
export compress_tensor_chain!, decompose_tensor!
export contract_mps_tensor_network_circuit!
export calculate_mps_amplitudes!
export cost

include("layer2/slicing.jl")
include("layer2/netcon_contraction.jl")
include("layer2/bgreedy_contraction.jl")
include("layer2/quickbb_contraction.jl")
# include("layer2/KaHyPar_contraction.jl")

using HDF5


function merge_common_bonds!(network::TensorNetworkCircuit, a_label::Symbol, b_label::Symbol)
    a = network.nodes[a_label]
    b = network.nodes[b_label]
    a_common_indices = findall(x -> x in b.indices, a.indices)
    common_edges = a.indices[a_common_indices]

    if length(a_common_indices) > 1

        index_map_a = Dict([v => k for (k, v) in enumerate(a.indices)])
        index_map_b = Dict([v => k for (k, v) in enumerate(b.indices)])

        a_remaining_indices = findall(x -> !(x in b.indices), a.indices)
        permute_tensor(network, a_label, vcat(a_remaining_indices, a_common_indices))

        b_common_indices = findall(x -> x in a.indices, b.indices)
        b_remaining_indices = findall(x -> !(x in a.indices), b.indices)
        permute_tensor(network, b_label, vcat(b_remaining_indices, b_common_indices))

        new_edge = new_label!(network, "index")
        indices_a = vcat(a.indices[a_remaining_indices], [new_edge])
        indices_b = vcat(b.indices[b_remaining_indices], [new_edge])

        dims_map_a = Dict{Symbol, Int64}(a.indices .=> a.dims)
        dims_map_a = Dict{Symbol, Int64}(b.indices .=> b.dims)
        dims_map_a[new_edge] = dims_map_b[new_edge] = prod([dims_map_a[ind] for ind in a_common_indices])
        dims_a = [dims_map_a[index] for index in indices_a]
        dims_b = [dims_map_b[index] for index in indices_b]

        network.nodes[a_label] = Node(indices_a, dims_a, a_label)
        network.nodes[b_label] = Node(indices_b, dims_b, b_label)

        l, m = length(a_remaining_indices), length(a_common_indices)
        groups = [map(x -> [x], 1:l)..., collect((l+1):l+m)]
        reshape_tensor(network, a_label, groups)

        l, m = length(b_remaining_indices), length(b_common_indices)
        groups = [map(x -> [x], 1:l)..., collect((l+1):l+m)]
        reshape_tensor(network, b_label, groups)

        network.edges[new_edge] = Edge(a_label, b_label, nothing, true)
        for common_edge in common_edges
            delete!(network.edges, common_edge)
        end
    end
end

"""
    function inorder_contraction!(network::TensorNetworkCircuit)

Function to contract the network in order starting from input nodes
"""
function inorder_contraction!(network::TensorNetworkCircuit)
    layer_nodes = Dict{Int64, Array{Symbol,1}}()
    for (k, v) in pairs(network.node_layers)
        if haskey(layer_nodes, v)
            push!(layer_nodes[v], k)
        else
            layer_nodes[v] = [k]
        end
    end

    gate_layers = [k for k in sort(collect(keys(layer_nodes))) if k > 0]
    if haskey(layer_nodes, -1)
        gate_layers = vcat(gate_layers, [-1])
    end

    for layer in gate_layers
        nodes = layer_nodes[layer]
        new_nodes = Array{Symbol, 1}(undef, length(nodes))
        for (i, node) in enumerate(nodes)
            in_nodes = unique(inneighbours(network, node))
            for in_node in in_nodes
                node = contract_pair!(network, in_node, node)
            end
            new_nodes[i] = node
        end
        # merge multiple virtual bonds if present
        for (x, y) in Iterators.product(new_nodes, new_nodes)
            if y > x
                merge_common_bonds!(network, x, y)
            end
        end
    end
end

"""
    function random_contraction_plan(network::TensorNetworkCircuit)

Function to create a random contraction plan
"""
function random_contraction_plan(network::TensorNetworkCircuit)
    closed_edges = [k for (k, v) in pairs(network.edges) if v.src !== nothing && v.dst !== nothing]
    shuffle(closed_edges)
end

"""
    function full_wavefunction_contraction!(network::TensorNetworkCircuit,
                                            output_shape::Union{String, Array{<:Integer, 1}}="")

Function to contract a network by first contracting input nodes together, to
get the wavefunction representing the initial state, and then contracting
gates into the wavefunction in the order they appear in the circuit.
"""
function full_wavefunction_contraction!(network::TensorNetworkCircuit,
                                        output_shape::Union{String, Array{<:Integer, 1}}="")

    # Get the input nodes of the circuit
    input_nodes = [network.edges[edge].src for edge in network.input_qubits]
    if nothing in input_nodes
        error("Please create nodes for each input before using wavefunction
               contraction")
    end

    # Contract all of the input nodes to get a tensor representing the full
    # initial wavefunction.
    wf = input_nodes[1]
    for wfi in input_nodes[2:end]
        wf = contract_pair!(network, wf, wfi)
    end

    layer_nodes = Dict{Int64, Array{Symbol,1}}()
    for (k, v) in pairs(network.node_layers)
        if haskey(layer_nodes, v)
            push!(layer_nodes[v], k)
        else
            layer_nodes[v] = [k]
        end
    end

    gate_layers = [k for k in sort(collect(keys(layer_nodes))) if k > 0]
    if haskey(layer_nodes, -1)
        gate_layers = vcat(gate_layers, [-1])
    end
    for layer in gate_layers
        nodes = layer_nodes[layer]

        # contract nodes in the layer and then contract with the wf vector
        n = nodes[1]
        if length(nodes) > 1
            for n2 in nodes[2:end]
                n = contract_pair!(network, n, n2)
            end
        end
        wf = contract_pair!(network, wf, n)
    end

    # Permute the indices of the final tensor to have the correct order.
    output_tensor = Symbol("node_$(network.counters["node"])")
    node = network.nodes[output_tensor]
    if length(node.indices) != 0
        order = [findfirst(x->x==ind, node.indices) for ind in network.output_qubits]
        node.indices[:] = network.output_qubits[:]
        permute_tensor(network, output_tensor, order)

        # Reshape the final tensor if a shape is specified by the user.
        if output_shape == "vector"
            reshape_tensor(network, output_tensor, [collect(1:length(node.indices))])
        elseif output_shape != ""
            reshape_tensor(network, output_tensor, output_shape)
        end
    end

    # save the final tensor under the name "result".
    save_output(network, output_tensor)
end


# *************************************************************************** #
#                  Functions to contract a tensor network
# *************************************************************************** #

"""
    function sort_indices(A::Node, B::Node)

This function divides all of the indices of two nodes A and B into two arrays,
one array for all the shared indices between A and B and a second array for all
the indices that are not shared.
"""
function sort_indices(A::Node, B::Node)
    common_indices = intersect(A.indices, B.indices)
    remaining_indices_A = setdiff(A.indices, common_indices)
    remaining_indices_B = setdiff(B.indices, common_indices)
    uncommon_indices = union(remaining_indices_A, remaining_indices_B)

    common_indices, uncommon_indices
end

"""
    function create_ncon_indices(A::Node, B::Node,
                                 common_indices::Array{Symbol, 1},
                                 uncommon_indices::Array{Symbol, 1})

This function returns ncon indices for the nodes A and B. These can then to be
passed to ncon in order to contract the tensor data associated with A and B.
"""
function create_ncon_indices(A::Node, B::Node,
                             common_indices::Array{Symbol, 1},
                             uncommon_indices::Array{Symbol, 1})
    # Contracted indices are assigned positive integers, uncontracted indices
    # are assigned negative integers.
    common_index_map = Dict([(x[2], x[1]) for x in enumerate(common_indices)])
    uncommon_indices_map = Dict([(x[2], -x[1]) for x
                                    in enumerate(uncommon_indices)])

    index_map = merge(common_index_map, uncommon_indices_map)
    A_ncon_indices = [index_map[x] for x in A.indices]
    B_ncon_indices = [index_map[x] for x in B.indices]

    return A_ncon_indices, B_ncon_indices
end

"""
    function contract_network!(network::TensorNetworkCircuit,
                               plan::Array{Symbol, 1},
                               output_shape::Union{String, Array{<:Array{<:Integer, 1}, 1})

Function to contract the given network according to the given contraction plan.
The resulting tensor will be given the shape described by 'output_shape'.
"""
function contract_network!(network::TensorNetworkCircuit,
                           plan::Array{Symbol, 1},
                           output_shape::Union{String, Array{<:Integer, 1}}="")

    # Loop through the plan and contract each edge in sequence
    for edge in plan
        contract_pair!(network, edge)
    end

    # Contract disjoint pieces of the network, if any
    while length(network.nodes) > 1
        n1, state = iterate(network.nodes)
        n2, _ = iterate(network.nodes, state)
        contract_pair!(network, n1.first, n2.first)
    end

    # Permute the indices of the final tensor to have the correct order.
    output_tensor = Symbol("node_$(network.counters["node"])")
    node = network.nodes[output_tensor]
    if length(node.indices) != 0
        order = [findfirst(x->x==ind, node.indices) for ind in network.output_qubits]
        node.indices[:] = network.output_qubits[:]
        permute_tensor(network, output_tensor, order)

        # Reshape the final tensor if a shape is specified by the user.
        if output_shape == "vector"
            reshape_tensor(network, output_tensor, [collect(1:length(node.indices))])
        elseif output_shape != ""
            reshape_tensor(network, output_tensor, output_shape)
        end
    end

    # save the final tensor under the name "result".
    save_output(network, output_tensor)
end

"""
    function contract_network!(network::TensorNetworkCircuit,
                               plan::Array{Array{Symbol, 1}, 1},
                               output_shape::Union{String, Array{<:Array{<:Integer, 1}, 1})

Function to contract the given network according to the given contraction plan.
The resulting tensor will be given the shape described by 'output_shape'.
"""
function contract_network!(network::TensorNetworkCircuit,
                           plan::Array{Array{Symbol, 1}, 1},
                           output_shape::Union{String, Array{<:Integer, 1}}="")

    # Loop through the plan and contract each tensor pair in sequence
    for (A, B) in plan
        contract_pair!(network, A, B)
    end

    # Permute the indices of the final tensor to have the correct order.
    output_tensor = Symbol("node_$(network.counters["node"])")
    node = network.nodes[output_tensor]
    if length(node.indices) != 0
        order = [findfirst(x->x==ind, node.indices) for ind in network.output_qubits]
        node.indices[:] = network.output_qubits[:]
        permute_tensor(network, output_tensor, order)

        # Reshape the final tensor if a shape is specified by the user.
        if output_shape == "vector"
            reshape_tensor(network, output_tensor, [collect(1:length(node.indices))])
        elseif output_shape != ""
            reshape_tensor(network, output_tensor, output_shape)
        end
    end

    # save the final tensor under the name "result".
    save_output(network, output_tensor)
end

"""
    function contract_pair!(network::TensorNetworkCircuit,
                            edge::Symbol)

Contract a pair of nodes connected by the given edge.
"""
function contract_pair!(network::TensorNetworkCircuit,
                        edge::Symbol)
    # sometimes edges get contracted ahead of time if connecting two
    # tensors being contracted
    if edge in keys(network.edges)
        e = network.edges[edge]
        return contract_pair!(network, e.src, e.dst)
    end
end

"""
    function contract_pair!(network::TensorNetworkCircuit,
                            A_label::Symbol,
                            B_label::Symbol)

Function to contract the nodes A and B of the network.
"""
function contract_pair!(network::TensorNetworkCircuit,
                        A_label::Symbol,
                        B_label::Symbol)

    # Should only contract different tensors, so skip contraction if A = B.
    if A_label == B_label
        return nothing
    end

    # Get the tensors being contracted and denote them A and B
    A = network.nodes[A_label]
    B = network.nodes[B_label]

    # Sort the indices of A and B into shared and unshared.
    common_indices, remaining_indices = sort_indices(A, B)

    # Create array of ncon indices for A and B.
    A_ncon_indices, B_ncon_indices = create_ncon_indices(A, B,
                                                         common_indices,
                                                         remaining_indices)


    # Get the dimensions of the contracted and open indices and use them
    # to record the costs of contraction
    dims_map = Dict{Symbol, Int}([A.indices; B.indices] .=> [A.dims; B.dims])
    contracted_dims = [dims_map[ind] for ind in common_indices]
    C_dims = [dims_map[ind] for ind in remaining_indices]
    record_compute_costs!(network.backend, C_dims, contracted_dims)

    # Create and add the new node to the tensor network.
    C_label = new_label!(network, "node")
    C = Node(remaining_indices, C_dims, C_label)
    network.nodes[C_label] = C

    # remove contracted edges
    for index in common_indices
        delete!(network.edges, index)
    end

    # replumb existing edges to point to new node
    # TODO: might become time consuming for large networks
    for index in remaining_indices
        e = network.edges[index]
        if e.src in [A_label, B_label]
            e.src = C_label
        elseif e.dst in [A_label, B_label]
            e.dst = C_label
        end
    end

    # delete the nodes that were contracted
    delete!(network.nodes, A_label)
    delete!(network.nodes, B_label)

    # Call lower level function to perform contraction on tensors
    contract_tensors(network, A_label, A_ncon_indices,
                     B_label, B_ncon_indices, C_label)

    C_label
end

# *************************************************************************** #
#                       Compression of a tensor network
# *************************************************************************** #

"""
    function compress_tensor_chain!(network::TensorNetworkCircuit,
                                    nodes::Array{Symbol, 1};
                                    threshold::AbstractFloat=1e-13,
                                    max_rank::Integer=0)

Compress a chain of tensors by given by the array of symbols. This is achieved
by peforming forward and backward sweeps where of compression operations on each
bond. Compression of each bond proceeds by discarding singular values and
corresponding axes with singular values below the given threshold or values
beyond the max_rank (max_rank zero corresponds to infinite rank)
"""
function compress_tensor_chain!(network::TensorNetworkCircuit,
                                nodes::Array{Symbol, 1};
                                threshold::AbstractFloat=1e-13,
                                max_rank::Integer=0)

    # TODO: required that tensors in the chain only have virtual bonds between
    # consecutive tensors. Add a check that enforces this

    # forward pass
    for i in 1:(length(nodes) - 1)
        left_node = network.nodes[nodes[i]]
        right_node = network.nodes[nodes[i+1]]

        left_indices = setdiff(left_node.indices, right_node.indices)
        right_indices = setdiff(right_node.indices, left_node.indices)

        combined_node = contract_pair!(network, nodes[i], nodes[i+1])

        decompose_tensor!(network, combined_node, left_indices, right_indices,
                          left_label=nodes[i],
                          right_label=nodes[i+1],
                          threshold=threshold,
                          max_rank=max_rank
                         )
    end

    # now do a backward pass
    for i in (length(nodes) - 1):-1:1
        left_node = network.nodes[nodes[i]]
        right_node = network.nodes[nodes[i+1]]

        left_indices = setdiff(left_node.indices, right_node.indices)
        right_indices = setdiff(right_node.indices, left_node.indices)

        combined_node = contract_pair!(network, nodes[i], nodes[i+1])

        decompose_tensor!(network, combined_node, left_indices, right_indices,
                          left_label=nodes[i],
                          right_label=nodes[i+1],
                          threshold=threshold,
                          max_rank=max_rank
                         )
    end

end

"""
    function decompose_tensor!(tng::TensorNetworkCircuit,
                               node::Symbol
                               left_indices::Array{Symbol, 1},
                               right_indices::Array{Symbol, 1};
                               threshold::AbstractFloat=1e-15,
                               max_rank::Integer=0,
                               left_label::Union{Nothing, Symbol}=nothing,
                               right_label::Union{Nothing, Symbol}=nothing)

Decompose a tensor into two smaller tensors
"""
function decompose_tensor!(network::TensorNetworkCircuit,
                           node_label::Symbol,
                           left_indices::Array{Symbol, 1},
                           right_indices::Array{Symbol, 1};
                           threshold::AbstractFloat=1e-14,
                           max_rank::Integer=0,
                           left_label::Union{Nothing, Symbol}=nothing,
                           right_label::Union{Nothing, Symbol}=nothing)

    node = network.nodes[node_label]
    index_map = Dict([v => k for (k, v) in enumerate(node.indices)])
    left_positions = [index_map[x] for x in left_indices]
    right_positions = [index_map[x] for x in right_indices]

    # plumb these nodes back into the graph and delete the original
    B_label = (left_label === nothing) ? new_label!(network, "node") : left_label
    C_label = (right_label === nothing) ? new_label!(network, "node") : right_label

    chi = decompose_tensor!(network,
                            node_label,
                            left_positions,
                            right_positions;
                            threshold=threshold,
                            max_rank=max_rank,
                            left_label=B_label,
                            right_label=C_label)

    # Work out the dimensions of the indices.
    B_dims = [node.dims[i] for i in left_positions]
    C_dims = [node.dims[i] for i in right_positions]
    if chi > 0
        virtual_dim = chi
    else
        left_dim = prod(B_dims)
        right_dim = prod(C_dims)
        virtual_dim = min(left_dim, right_dim)
        if max_rank > 0
            virtual_dim = min(virtual_dim, max_rank)
        end
    end
    append!(B_dims, virtual_dim)
    prepend!(C_dims, virtual_dim)

    index_label = new_label!(network, "index")
    B_node = Node(vcat(left_indices, [index_label,]), B_dims, B_label)
    C_node = Node(vcat([index_label,], right_indices), C_dims, C_label)

    # add the nodes
    network.nodes[B_label] = B_node
    network.nodes[C_label] = C_node

    # remap edge endpoints
    for index in left_indices
        if network.edges[index].src == node_label
            network.edges[index].src = B_label
        elseif network.edges[index].dst == node_label
            network.edges[index].dst = B_label
        end
    end
    for index in right_indices
        if network.edges[index].src == node_label
            network.edges[index].src = C_label
        elseif network.edges[index].dst == node_label
            network.edges[index].dst = C_label
        end
    end

    # add new edge
    network.edges[index_label] = Edge(B_label, C_label, nothing, true)

    delete!(network.nodes, node_label)

    (B_label, C_label)
end

"""
    function contract_mps_tensor_network_circuit(network::TensorNetworkCircuit;
                                                 max_bond::Integer=2,
                                                 threshold::AbstractFloat=1e-13,
                                                 max_rank::Integer=0)

Contract a tensor network representing a quantum circuit using MPS techniques
"""
function contract_mps_tensor_network_circuit!(network::TensorNetworkCircuit;
                                              max_bond::Integer=2,
                                              threshold::AbstractFloat=1e-13,
                                              max_rank::Integer=0)
    # identify the MPS nodes as the input nodes
    mps_nodes = [network.edges[x].src for x in network.input_qubits]
    @assert all([x !== nothing for x in mps_nodes]) "Input qubit values must be set"

    # identify the MPS nodes as the input nodes
    qubits = length(network.input_qubits)
    mps_nodes = [network.edges[x].src for x in network.input_qubits]
    @assert all([x !== nothing for x in mps_nodes]) "Input qubit values must be set"

    layer_nodes = Dict{Int64, Array{Symbol,1}}()
    for (k, v) in pairs(network.node_layers)
        if haskey(layer_nodes, v)
            push!(layer_nodes[v], k)
        else
            layer_nodes[v] = [k]
        end
    end

    gate_layers = [k for k in sort(collect(keys(layer_nodes))) if k > 0]
    if haskey(layer_nodes, -1)
        gate_layers = vcat(gate_layers, [-1])
    end

    # gate_nodes = setdiff(collect(keys(network.nodes)), mps_nodes)
    bond_counts = zeros(Int, qubits-1)
    for gate_layer in gate_layers
        nodes = layer_nodes[gate_layer]

        updated_indices = Array{Int64, 1}(undef, length(nodes))
        for (i, node) in enumerate(nodes)
            # find the input node to connect to it
            input_node = inneighbours(network, node)[1]
            @assert input_node in mps_nodes "$input_node not in mps nodes"

            idx = findfirst(x -> x == input_node, mps_nodes)
            mps_nodes[idx] = contract_pair!(network, input_node, node)
            updated_indices[i] = idx
         end

        sort!(updated_indices)
        for i in 1:(length(updated_indices)-1)
            @assert (updated_indices[i+1] -  updated_indices[i]) == 1 "Gates between non-neighboring qubits"
            bond_counts[updated_indices[i]] += 1
        end

        if maximum(bond_counts) > max_bond
            compress_tensor_chain!(network, mps_nodes, threshold=threshold, max_rank=max_rank)
            bond_counts[:] .= 0
        end
    end
    compress_tensor_chain!(network, mps_nodes, threshold=threshold, max_rank=max_rank)

    # save the final mps tensors
    for (i, node) in enumerate(mps_nodes)
        save_output(network, node, String(node))
    end
    mps_nodes
end

"""
    function calculate_mps_amplitudes!(network::TensorNetworkCircuit,
                                      mps_nodes::Array{Symbol, 1},
                                      result::String="result")

Calculate amplitudes from an MPS state
"""
function calculate_mps_amplitudes!(network::TensorNetworkCircuit,
                                   mps_nodes::Array{Symbol, 1},
                                   result::String="result")
    output_node = mps_nodes[1]
    for node in mps_nodes[2:end]
        output_node = contract_pair!(network, output_node, node)
    end
    permute_tensor(network, output_node, network.qubit_ordering)
    reshape_tensor(network, output_node, [collect(1:length(mps_nodes))])
    save_output(network, output_node, result)
end
