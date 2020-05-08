export contract_edge!, contract_network!

using TensorOperations

"""
    function contract_edge!(network::TensorNetworkCircuit,
                            A::Symbol, B::Symbol)

Contract the edges that connect nodes A and B
"""
function contract_edge!(network::TensorNetworkCircuit,
                        A_label::Symbol,
                        B_label::Symbol)
    # Should only contract different tensors so skip contraction if A = B
    if A_label == B_label
        return nothing
    end

    # Get the tensors being contracted and denote them A and B
    A = network.nodes[A_label]
    B = network.nodes[B_label]

    # Create an array of index labels for the new node
    common_indices = intersect(A.indices, B.indices)
    remaining_indices_A = setdiff(A.indices, common_indices)
    remaining_indices_B = setdiff(B.indices, common_indices)
    remaining_indices = union(remaining_indices_A, remaining_indices_B)

    common_index_map = Dict([(x[2], x[1]) for x in enumerate(common_indices)])
    remaining_indices_map = Dict([(x[2], -x[1]) for x
                                    in enumerate(remaining_indices)])
    index_map = merge(common_index_map, remaining_indices_map)
    A_ncon_indices = [index_map[x] for x in A.indices]
    B_ncon_indices = [index_map[x] for x in B.indices]

    # contract the tensors
    C_data = ncon((A.data, B.data), (A_ncon_indices, B_ncon_indices))
    if typeof(C_data) <: Number
        C_data = [C_data]
    end

    # create the new node
    C = Node(remaining_indices, C_data)
    C_label = new_label!(network, "node")
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
end

"""
    function contract_edge!(network::TensorNetworkCircuit,
                            edge::Symbol)

Contract the edge with given label
"""
function contract_edge!(network::TensorNetworkCircuit,
                        edge::Symbol)
    if edge in keys(network.edges)
        e = network.edges[edge]
        contract_edge!(network, e.src, e.dst)
    end
end

"""
    function contract_network(network::TensorNetworkCircuit,
                              plan::Array{Array{Int, 1}, 1})

Contract the network according to the provided contraction plan
"""
function contract_network!(network::TensorNetworkCircuit,
                           plan::Array{Symbol, 1})
    # Loop through the plan and contract each edge in sequence
    for edge in plan
        # sometimes edges get contracted ahead of time if connecting two
        # tensors being contracted
        if edge in keys(network.edges)
            contract_edge!(network, edge)
        end
    end

    # Contract disjoint pieces of the network, if any
    while length(network.nodes) > 1
        n1, state = iterate(network.nodes)
        n2, _ = iterate(network.nodes, state)
        contract_edge!(network, n1.first, n2.first)
    end
end
