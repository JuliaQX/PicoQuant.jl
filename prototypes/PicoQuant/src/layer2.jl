import Random.shuffle

export random_contraction_plan
export contraction_plan_to_json, contraction_plan_from_json
export contract_pair!, contract_network!
export compress_tensor_chain!

using HDF5

"""
    function random_contraction_plan(network::TensorNetworkCircuit)

Function to create a random contraction plan
"""
function random_contraction_plan(network::TensorNetworkCircuit)
    closed_edges = [x for (x, y) in pairs(edges(network)) if y.src != nothing
                                                          && y.dst != nothing]
    shuffle(closed_edges)
end

# function cost_flops(network, plan)
# end
#
# function cost_max_memory(network, plan)
# end
#
# function find_contraction_plan(network, cost_function)
# end


"""
    function contraction_plan_to_json(plan::Array)

Function to serialise the contraction plan to json format
"""
function contraction_plan_to_json(plan::Array{Symbol})
    JSON.json(plan)
end

"""
    function contraction_plan_from_json(str::String)

Function to deserialize the contraction plan from a json string
"""
function contraction_plan_from_json(str::String)
    [Symbol(x) for x in JSON.parse(str)]
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
                               output_shape::Union{String, Array{<:Integer, 1}})

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

    # Reshape the final tensor if a shape is specified by the user.
    output_tensor = Symbol("node_$(network.counters["node"])")
    if output_shape == "vector"
        vector_length = 2^length(network.output_qubits)
        reshape_tensor(backend, output_tensor, vector_length)

    elseif output_shape != ""
        reshape_tensor(backend, output_tensor, output_shape)
    end

    # save the final tensor under the name "result".
    save_output(backend, output_tensor)
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

    # Create and add the new node to the tensor network.
    C_label = new_label!(network, "node")
    C = Node(remaining_indices, C_label)
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

    # Get the backend to contract the tensors
    contract_tensors(backend, A_label, A_ncon_indices,
                     B_label, B_ncon_indices, C_label)

    C_label
end

# *************************************************************************** #
#                       Compression of a tensor network
# *************************************************************************** #

"""
    function compress_tensor_chain!(tng::TensorNetworkCircuit,
                                    nodes::Symbol;
                                    threshold::AbstractFloat=1e-15)

Compress a chain of tensors
"""
function compress_tensor_chain!(tng::TensorNetworkCircuit,
                                nodes::Array{Symbol, 1};
                                threshold::AbstractFloat=1e-15)

    # forward pass
    for i in 1:(length(nodes) - 1)
        left_node = tng.nodes[nodes[i]]
        right_node = tng.nodes[nodes[i+1]]

        left_indices = setdiff(left_node.indices, right_node.indices)
        right_indices = setdiff(right_node.indices, left_node.indices)

        combined_node = contract_pair!(tng, nodes[i], nodes[i+1])

        decompose_tensor!(tng, combined_node, left_indices, right_indices,
                          left_label=nodes[i],
                          right_label=nodes[i+1]
                         )
    end

    # now do a backward pass
    for i in (length(nodes) - 1):-1:1
        left_node = tng.nodes[nodes[i]]
        right_node = tng.nodes[nodes[i+1]]

        left_indices = setdiff(left_node.indices, right_node.indices)
        right_indices = setdiff(right_node.indices, left_node.indices)

        combined_node = contract_pair!(tng, nodes[i], nodes[i+1])

        decompose_tensor!(tng, combined_node, left_indices, right_indices,
                          left_label=nodes[i],
                          right_label=nodes[i+1]
                         )
    end

end
