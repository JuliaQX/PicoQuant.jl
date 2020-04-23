export contract_edge!, contract_network!

"""
    function contract_edge!(network::TensorNetworkCircuit,
                            A::Integer, B::Integer)

Contract the edge of the give tensor network that connects the tensor at
position A to the tensor at position B
"""
function contract_edge!(network::TensorNetworkCircuit, A_position, B_position)
    # Get the tensors being contracted and denote them A and B
    A = network.nodes[A_position]
    B = network.nodes[B_position]

    # Create an array of index labels for the new node
    common_indices = intersect(A.indices,B.indices)
    lft_indices = setdiff(A.indices, common_indices)
    rgt_indices = setdiff(B.indices, common_indices)
    new_indices = union(lft_indices, rgt_indices)

    # Find the index permutation after the common indices in A have been moved
    # to the right for contraction
    ind_order_A = [findfirst(x->x==ind, A.indices)
                   for ind in [lft_indices; common_indices]]

    # Find the index permutation after the common indices in B have been moved
    # to the left for contraction
    ind_order_B = [findfirst(x->x==ind, B.indices)
                   for ind in [common_indices; rgt_indices]]

    # Permute the A and B tensors according to the shuffled index labels
    if length(ind_order_A) > 0 && length(ind_order_B) > 0
        A.data[:] = permutedims(A.data, ind_order_A)[:]
        B.data[:] = permutedims(B.data, ind_order_B)[:]
    end

    # Reshape the A and B tensors for matrix multiplication.
    m = length(lft_indices); i = length(common_indices); n = length(rgt_indices)
    A_data = reshape(A.data, 2^m, 2^i)
    B_data = reshape(B.data, 2^i, 2^n)

    # Contract A and B and reshape the result (assuming all bond dim = 2)
    data = A_data * B_data
    data = reshape(data, 2*Int.(ones(length(new_indices)))...)

    # Create a new Node for the contracted tensor
    new_node = Node(new_indices, data)

    # Replace tensors A and B in the network with the new tensor AB
    positions = findall(x->x==A || x==B, network.nodes)
    for i in positions
        network.nodes[i] = new_node
    end
end

"""
    function contract_network(network::TensorNetworkCircuit,
                              path::Array{Array{Int, 1}, 1})

Contract the network according to the provided contraction path
"""
function contract_network!(network::TensorNetworkCircuit,
                           path::Array{<:Array{<:Integer, 1}, 1})
    # Loop through the path and contract each edge in sequence
    for edge in path
        contract_edge!(network, edge[1], edge[2])
    end

    # Contract disjoint pieces of the network, if any
    num_qubits = length(network.output_qubits)
    for i = 2:num_qubits
        if network.nodes[1] != network.nodes[i]
            contract_edge!(network, 1, i)
        end
    end
end
