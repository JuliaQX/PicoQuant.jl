
export multi_index_partition, partition_network_on_virtual_bonds
export replace_with_view!, slice_tensor_network

"""
    function multi_index_partition(dims::Tuple{Vararg{Int64, N}},
                                   number_partitions::Integer,
                                   partition::Integer) where N

Find the starting and finishing index for the given dimension
"""
function multi_index_partition(dims::Tuple{Vararg{Int64, N}},
                               number_partitions::Integer,
                               partition::Integer) where N
    total_dim, num_dims = 1, 1
    while num_dims < length(dims)
        total_dim *=dims[num_dims]
        if total_dim == number_partitions
            break
        elseif total_dim > number_partitions
            @error "Partitions and product of dimensions must match"
        end
        num_dims += 1
    end
    CartesianIndices(dims[1:num_dims])[partition]
end


"""
    function partition_network_on_virtual_bonds(tn::TensorNetworkCircuit,
                                                number_partitions::Int64,
                                                partition::Int64)

Given a tensor network circuit, find the virtual bond to slice and values
for a given partition
"""
function partition_network_on_virtual_bonds(network::TensorNetworkCircuit,
                                            number_partitions::Int64,
                                            partition::Int64)
    # find the virtual bonds
    virtual_bonds = [e for e in edges(network) if e[2].virtual]

    # find virtual bond dimensions
    bond_dims = Array{Int64, 1}(undef, length(virtual_bonds))  # preallocate
    for (i, (edge_index, edge)) in enumerate(virtual_bonds)
        node_data = load_tensor_data(network, edge.src)
        node_indices = findall(x -> x == edge_index, network.nodes[edge.src].indices)
        bond_dims[i] = prod(size(node_data)[node_indices])
    end
    bond_order = sortperm(bond_dims)
    virtual_bonds = [e[1] for e in virtual_bonds[bond_order]]
    bond_dims = bond_dims[bond_order]

    @assert length(bond_dims) > 0 "There must be some virtual bonds, try turning on decompose"
    ci = multi_index_partition(Tuple(bond_dims), number_partitions, partition)
    virtual_bonds[1:length(ci)], ci
end

"""
    function replace_with_view!(network::TensorNetworkCircuit,
                                node_label::Symbol,
                                bond_label::Symbol,
                                bond_range::UnitRange{<:Integer})

Replace the node in the network with a view on the network along the provided bond ranges
"""
function replace_with_view!(network::TensorNetworkCircuit,
                            node_label::Symbol,
                            bond_label::Symbol,
                            bond_range::UnitRange{<:Integer})
    # find the index of the node to create view on
    node = network.nodes[node_label]
    bond_idx = findfirst(x -> x == bond_label, node.indices)
    view_node = new_label!(network, "node")

    # plumb in this view node in place of original
    network.nodes[view_node] = Node(node.indices, node.dims, view_node)
    network.node_layers[view_node] = network.node_layers[node_label]

    for edge_index in node.indices
        edge = network.edges[edge_index]
        if edge.src == node_label
            edge.src = view_node
        else
            edge.dst = view_node
        end
    end

    view_tensor!(network, view_node, node_label, bond_idx, bond_range)
    delete_tensor!(network, node_label)
    delete!(network.nodes, node_label)
    delete!(network.node_layers, node_label)
end

"""
function slice_tensor_network(network::TensorNetworkCircuit,
                              bond_labels::Array{Symbol, 1},
                              bond_values::CartesianIndex)
"""
function slice_tensor_network(network::TensorNetworkCircuit,
                              bond_labels::Array{Symbol, 1},
                              bond_values::CartesianIndex)
    for (bond_label, bond_value) in zip(bond_labels, Tuple(bond_values))
        # find the node labels the bond connects to
        node_1 = network.edges[bond_label].src
        node_2 = network.edges[bond_label].dst
        replace_with_view!(network, node_1, bond_label, bond_value:bond_value)
        replace_with_view!(network, node_2, bond_label, bond_value:bond_value)
    end
end
