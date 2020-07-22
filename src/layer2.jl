import Random.shuffle

export random_contraction_plan, inorder_contraction_plan
export contraction_plan_to_json, contraction_plan_from_json
export contract_pair!, contract_network!, full_wavefunction_contraction!
export compress_tensor_chain!, decompose_tensor!
export contract_mps_tensor_network_circuit!
export calculate_mps_amplitudes!

using HDF5

"""
    function inorder_contraction_plan(network::TensorNetworkCircuit)

Function to return contraction plan by order of indices
"""
function inorder_contraction_plan(network::TensorNetworkCircuit)
    [x for (x, y) in pairs(edges(network)) if y.src != nothing
                                           && y.dst != nothing]
end

"""
    function random_contraction_plan(network::TensorNetworkCircuit)

Function to create a random contraction plan
"""
function random_contraction_plan(network::TensorNetworkCircuit)
    closed_edges = inorder_contraction_plan(network)
    shuffle(closed_edges)
end

"""
    function full_wavefunction_contraction!(tn::TensorNetworkCircuit,
                                            output_shape::Union{String, Array{<:Integer, 1}}="")

Function to contract a network by first contracting input nodes together, to
get the wavefunction representing the initial state, and then contracting
gates into the wavefunction in the order they appear in the circuit.
"""
function full_wavefunction_contraction!(tn::TensorNetworkCircuit,
                                        output_shape::Union{String, Array{<:Integer, 1}}="")

    # Get the input nodes of the circuit
    input_nodes = [tn.edges[edge].src for edge in tn.input_qubits]
    if nothing in input_nodes
        error("Please create nodes for each input before using wavefunction
               contraction")
    end

    # Contract all of the input nodes to get a tensor representing the full
    # initial wavefunction.
    wf = input_nodes[1]
    for wfi in input_nodes[2:end]
        wf = contract_pair!(tn, wf, wfi)
    end

    # While there's more than one node in the network, contract all other nodes
    # connected to the wavefunction into the wavefunction node.
    while length(tn.nodes) > 1

        # Contract the wavefunction with each of its neighbours.
        for neighbour in outneighbours(tn, wf)

            # Skip this node if it was already contracted into the wavefunction
            # (and so no longer in tn.nodes). This happens when multiple edges
            # point from the wavefunction to the same node.
            if !(neighbour in keys(tn.nodes))
                continue
            end
            neighbour_node = tn.nodes[neighbour]

            # If the next node to be contracted into the wavefunction has an
            # edge whose src is something other then the wf or itself, then
            # some other node of the circuit should be contracted into wf before
            # this one.
            skip = false
            for index in neighbour_node.indices
                e = tn.edges[index]
                if !(e.src in [wf, neighbour])
                    skip = true
                    break
                end
            end
            if !skip
                wf = contract_pair!(tn, wf, neighbour)
            end
        end
    end


    # Permute the indices of the final tensor to have the correct order.
    output_tensor = Symbol("node_$(tn.counters["node"])")
    node = tn.nodes[output_tensor]
    if length(node.indices) != 0
        order = [findfirst(x->x==ind, node.indices) for ind in tn.output_qubits]
        node.indices[:] = tn.output_qubits[:]
        permute_tensor(backend, output_tensor, order)

        # Reshape the final tensor if a shape is specified by the user.
        if output_shape == "vector"
            vector_length = 2^length(tn.output_qubits)
            reshape_tensor(backend, output_tensor, vector_length)
        elseif output_shape != ""
            reshape_tensor(backend, output_tensor, output_shape)
        end
    end

    # save the final tensor under the name "result".
    save_output(backend, output_tensor)
end


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
    function compress_tensor_chain!(network::TensorNetworkCircuit,
                                    nodes::Array{Symbol, 1};
                                    threshold::AbstractFloat=1e-13)

Compress a chain of tensors by given by the array of symbols. This is achieved
by peforming forward and backward sweeps of compressing each bond.
"""
function compress_tensor_chain!(network::TensorNetworkCircuit,
                                nodes::Array{Symbol, 1};
                                threshold::AbstractFloat=1e-13)

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
                          threshold=threshold
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
                          threshold=threshold
                         )
    end

end

"""
    function decompose_tensor!(tng::TensorNetworkCircuit,
                               node::Symbol
                               left_indices::Array{Symbol, 1},
                               right_indices::Array{Symbol, 1};
                               threshold::AbstractFloat=1e-15,
                               left_label::Union{Nothing, Symbol}=nothing,
                               right_label::Union{Nothing, Symbol}=nothing)

Decompose a tensor into two smaller tensors
"""
function decompose_tensor!(tng::TensorNetworkCircuit,
                           node_label::Symbol,
                           left_indices::Array{Symbol, 1},
                           right_indices::Array{Symbol, 1};
                           threshold::AbstractFloat=1e-14,
                           left_label::Union{Nothing, Symbol}=nothing,
                           right_label::Union{Nothing, Symbol}=nothing)

    node = tng.nodes[node_label]
    index_map = Dict([v => k for (k, v) in enumerate(node.indices)])
    left_positions = [index_map[x] for x in left_indices]
    right_positions = [index_map[x] for x in right_indices]

    # plumb these nodes back into the graph and delete the original
    B_label = (left_label == nothing) ? new_label!(tng, "node") : left_label
    C_label = (right_label == nothing) ? new_label!(tng, "node") : right_label
    index_label = new_label!(tng, "index")
    B_node = Node(vcat(left_indices, [index_label,]), B_label)
    C_node = Node(vcat([index_label,], right_indices), C_label)

    decompose_tensor!(backend,
                      node_label,
                      left_positions,
                      right_positions;
                      threshold=threshold,
                      left_label=B_label,
                      right_label=C_label)

    # add the nodes
    tng.nodes[B_label] = B_node
    tng.nodes[C_label] = C_node

    # remap edge endpoints
    for index in left_indices
        if tng.edges[index].src == node_label
            tng.edges[index].src = B_label
        elseif tng.edges[index].dst == node_label
            tng.edges[index].dst = B_label
        end
    end
    for index in right_indices
        if tng.edges[index].src == node_label
            tng.edges[index].src = C_label
        elseif tng.edges[index].dst == node_label
            tng.edges[index].dst = C_label
        end
    end

    # add new edge
    tng.edges[index_label] = Edge(B_label, C_label, nothing, true)

    delete!(tng.nodes, node_label)

    (B_label, C_label)
end

"""
    function contract_mps_tensor_network_circuit(network::TensorNetworkCircuit;
                                                 threshold::AbstractFloat=1e-13)

Contract a tensor network representing a quantum circuit using MPS techniques
"""
function contract_mps_tensor_network_circuit!(network::TensorNetworkCircuit;
                                              max_bond::Integer=2,
                                              threshold::AbstractFloat=1e-13)
    # identify the MPS nodes as the input nodes
    mps_nodes = [network.edges[x].src for x in network.input_qubits]
    @assert all([x != nothing for x in mps_nodes]) "Input qubit values must be set"

    # identify the MPS nodes as the input nodes
    qubits = length(network.input_qubits)
    mps_nodes = [network.edges[x].src for x in network.input_qubits]
    @assert all([x != nothing for x in mps_nodes]) "Input qubit values must be set"

    gate_nodes = setdiff(collect(keys(network.nodes)), mps_nodes)
    bond_counts = zeros(Int, qubits-1)
    for node in gate_nodes
        # find the input node to connect to it
        input_node = inneighbours(network, node)[1]
        @assert input_node in mps_nodes "$input_node not in mps nodes"
        idx = findfirst(x -> x == input_node, mps_nodes)
        if length(virtualneighbours(network, node)) > 0
            for vn in virtualneighbours(network, node)
                if vn in mps_nodes
                    vn_idx = findfirst(x -> x == vn, mps_nodes)
                    bond_counts[min(vn_idx, idx)] += 1
                end
            end
        end
        mps_nodes[idx] = contract_pair!(network, input_node, node)
        if maximum(bond_counts) > max_bond
            compress_tensor_chain!(network, mps_nodes)
            bond_counts[:] .= 0
        end
    end
    compress_tensor_chain!(network, mps_nodes)

    # save the final mps tensors
    for (i, node) in enumerate(mps_nodes)
        save_output(backend, node, String(node))
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
    permute_tensor(backend, output_node, network.qubit_ordering)
    reshape_tensor(backend, output_node, 2^length(mps_nodes))
    save_output(backend, output_node, result)
end
