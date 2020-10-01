using KaHyPar
using SparseArrays
using DataStructures

export KaHyPar_contraction_plan, KaHyPar_contraction!

# *************************************************************************** #
#                             KaHyPar functions
# *************************************************************************** #

"""
    function KaHyPar_contraction_plan(network::TensorNetworkCircuit,
                                      k::Int, ϵ::Real, V::Int)

This function uses KaHyPar to partition a TensorNetworkCircuit into k parts of
roughly equal size (how roughly is determined by ϵ) and then finds a contraction
plan for the partitions. This function is then recursively called on the
partitions to compute a complete contraction plan for the whole Tensor Network.
If the number of nodes in the network is lower than the threshold V then netcon
is used to find the optimal contraction plan.

network - The tensor network to build a contraction plan for.

k - The number of partitions to divide the network into at each recursive step.

ϵ - Imbalance parameter for KaHyPar

V - If the number of nodes in the given network is below this threshold then
    netcon is used to find the optimal contraction plan for it.
"""
function KaHyPar_contraction_plan(network::TensorNetworkCircuit,
                                  k::Int=2, ϵ::Real=0.1, V::Int=20,
                                  index_dims::Union{Dict{Symbol, Int}, Nothing}=nothing)

    # TODO: This won't be needed when the bgreedy method is added as an option
    # for finding contraction plans
    if k > 36
        error("The number of partitions should be less than 36 for netcon")
    end

    # Base case: If the number of nodes is less than the threshold V
    # use netcon to get the optimal contraction plan.
    if length(network.nodes) < V
        return netcon(network, :tree)
    end

    # If index_dims dictionary not given then create one. This is used to create
    # the network of partitions with the correct dimensions for edges connecting
    # different partitions (needed by netcon to find the contraction plan).
    if index_dims === nothing
        index_dims = Dict{Symbol, Int}()
        for node in values(network.nodes)
            for (index, dim) in zip(node.indices, node.dims)
                index_dims[index] = dim
            end
        end
    end
    # TODO: something to think about/consider:
    # The above could be made more efficient or avoided if dims were matched
    # with edge indices in the data structure somehow. eg if dims were stored
    # in a dictionary, mapping edges to dims, inside the node struct instead of
    # an array or if dims were stored in the edge structs.

    # Use KaHyPar to find k partitions of the tensor network.
    A = incidence_matrix(network)
    h = KaHyPar.HyperGraph(A)
    partitions = KaHyPar.partition(h, k, configuration = :edge_cut)

    # Find a contraction tree for the partitions.
    partitioned_net = network_of_partitions(network, partitions, index_dims)
    contraction_plan = netcon(partitioned_net, :tree)

    # Create a dictionary mapping partitions to subnetworks of the Tensor
    # Network. These subnetworks are to be partitioned in the next recursive
    # step.
    subnets = Dict{Symbol, TensorNetworkCircuit}()
    nodes = [keys(network.nodes)...]
    for i = 0:maximum(partitions)
        partition_i = findall(x -> x==i, partitions)
        label = Symbol("partition_$i")
        subnets[label] = TensorNetworkCircuit(network, nodes[partition_i])
    end

    # Recursively call KaHyPar_contraction_plan on the next layer of the
    # contraction tree.
    contraction_tree = Array{Any, 1}(undef, 2)
    contraction_tree[1] = KaHyPar_contraction_plan(contraction_plan[1],
                                                   subnets, k, ϵ, V, index_dims)
    contraction_tree[2] = KaHyPar_contraction_plan(contraction_plan[2],
                                                   subnets, k, ϵ, V, index_dims)

    contraction_tree
end

"""
    function KaHyPar_contraction_plan(contraction_step::Array{Any, 1},
                                      subnets::Dict{Symbol, TensorNetworkCircuit},
                                      k::Int, ϵ::Real, V::Int)

While using KaHyPar_contraction_plan to find a contraction plan for a network,
if an element of the next layer of a contraction tree happens to be a
contraction of two other objects, this function recursively calls
KaHyPar_contraction_plan on both of those elements.
"""
function KaHyPar_contraction_plan(contraction_step::Array{Any, 1},
                                  subnets::Dict{Symbol, TensorNetworkCircuit},
                                  k::Int, ϵ::Real, V::Int,
                                  index_dims::Dict{Symbol, Int})
    contraction_plan[1] = KaHyPar_contraction_plan(contraction_plan[1],
                                                   subnets, k, ϵ, V, index_dims)
    contraction_plan[2] = KaHyPar_contraction_plan(contraction_plan[2],
                                                   subnets, k, ϵ, V, index_dims)

    contraction_plan
end

"""
    function KaHyPar_contraction_plan(partition::Symbol,
                                      subnets::Dict{Symbol, TensorNetworkCircuit},
                                      k::Int, ϵ::Real, V::Int)

While using KaHyPar_contraction_plan to find a contraction plan for a network,
if an element of the next layer of a contraction tree is a symbol representing
a partition of the original tneor network, this function calls
KaHyPar_contraction_plan on the corresponding subnetwork.
"""
function KaHyPar_contraction_plan(partition::Symbol,
                                  subnets::Dict{Symbol, TensorNetworkCircuit},
                                  k::Int, ϵ::Real, V::Int,
                                  index_dims::Dict{Symbol, Int})
    KaHyPar_contraction_plan(subnets[partition], k, ϵ, V, index_dims)
end

"""
    function incidence_matrix(tn::TensorNetworkCircuit)

Create an incidence matrix for the given tensor network.
"""
function incidence_matrix(tn::TensorNetworkCircuit)
    vertices, edges = [], []

    for (node_label, node) in tn.nodes
        node_number = parse(Int, String(node_label)[6:end])
        for edge_label in node.indices
            edge_number = parse(Int, String(edge_label)[7:end])

            append!(vertices, node_number)
            append!(edges, edge_number)
        end
    end

    weigths = ones(Int, length(vertices))
    sparse(vertices, edges, weigths)
end


# TODO: The following function should probably be in layer3. However, a general
# 'subnetwork function' could return a network independent of the parent network
# (i.e. so changes to the subnetwork are not reflected in the parent network and
# should probably have its own backend etc.) which would be inefficient for this
# particular use case. Shoud look at other possible use cases before deciding
# on the best implementation.
"""
    function TensorNetworkCircuit(tn::TensorNetworkCircuit,
                                  nodes::Array{Symbol, 1})

Outer constructor to create an instance of TensorNetworkCircuit representing
a sub network of a given TensorNetworkCircuit. The given array of nodes are
copied from the original network to create the subnetwork.
"""
function TensorNetworkCircuit(tn::TensorNetworkCircuit, nodes::Array{Symbol, 1})

    # Copy the given nodes from the given tensor network.
    subnet_nodes = OrderedDict{Symbol, Node}(node_label => tn.nodes[node_label]
                   for node_label in nodes)

    # Copy the edges connected to the copied nodes.
    subnet_edges = OrderedDict{Symbol, Edge}()
    for node in values(subnet_nodes)
        for index in node.indices
            index in keys(subnet_edges) && continue
            edge = deepcopy(tn.edges[index])
            if !(edge.dst in nodes); edge.dst = nothing end
            if !(edge.src in nodes); edge.src = nothing end
            subnet_edges[index] = edge
        end
    end

    # Return a TensorNetworkCircuit representing the subnetwork.
    TensorNetworkCircuit(tn.backend, tn.number_qubits,
                         tn.input_qubits, tn.output_qubits,
                         subnet_nodes, subnet_edges,
                         tn.qubit_ordering, tn.counters, tn.node_layers)
end

"""
    function network_of_partitions(tn::TensorNetworkCircuit,
                                   partitioning::Array{Int, 1})

Given a Tensor Network and a partitioning of it, this function returns
a tensor network whose nodes represent the partitions and whose edges
correspond to the edges connecting partitions.
"""
function network_of_partitions(network::TensorNetworkCircuit,
                               partitions::Array{Int, 1},
                               index_dims::Dict{Symbol, Int})

    # Get the number of partitions and create a dictionary mapping nodes
    # to the partition to which they belong. (this construction depends
    # on tn.nodes being an ordered dictionary.)
    num_nodes = maximum(partitions)
    node_partition = Dict{Symbol, Int}(keys(network.nodes) .=> partitions)

    # Create a node for each partition.
    nodes = OrderedDict{Symbol, Node}()
    for i = 0:num_nodes
        label = Symbol("partition_$i")
        nodes[label] = Node(label)
    end

    # Create an edge for any edge of the original tensor network that connects
    # two partitions, or is an open edge connected to a particular partition.
    edges = OrderedDict{Symbol, Edge}()
    for (index, edge) in network.edges
        src, dst = edge.src, edge.dst
        src === nothing || (src = node_partition[src])
        dst === nothing || (dst = node_partition[dst])
        src == dst && continue # Skip this edge if it is a loop.

        if !(src === nothing)
            src = Symbol("partition_$src")
            append!(nodes[src].indices, [index])
            append!(nodes[src].dims, [index_dims[index]])
        end

        if !(dst === nothing)
            dst = Symbol("partition_$dst")
            append!(nodes[dst].indices, [index])
            append!(nodes[dst].dims, [index_dims[index]])
        end

        edges[index] = Edge(src, dst, edge.qubit, edge.virtual)
    end

    TensorNetworkCircuit(network.backend, 0, [], [], nodes, edges,
                         [], network.counters, network.node_layers)
end

"""
    function convert_tree_to_plan(tree::Union{Array{<:Any, 1}, Integer},
                                  last_node_label::Integer,
                                  tensor_labels::Array{Symbol, 1},
                                  plan::Array{Array{Symbol, 1}, 1})

This function converts a contraction tree, found by the netcon method, to a
contraction plan (array of node-label pairs) for PicoQuant contract functions.

tree - The contraction tree found by netcon.

last_node_label - Initially, this should be the number used to create the last
                  node added to the network being contracted, it is subsequantly
                  incremented to account for nodes created during contraction.

converted_plan - Initially, this argument should be an empty array. It will hold
                 the sequence of pairs when the function completes.
"""
function convert_tree_to_plan(tree::Union{Array{<:Any, 1}, Symbol},
                              last_node_label::Integer,
                              plan::Array{Array{Symbol, 1}, 1}
                                    =Array{Array{Symbol, 1}, 1}())

    # Recursively convert the plan into a sequence of pairs.
    if typeof(tree) == Symbol
        # Base case: If the plan is a Symbol corresponding to a single tensor,
        # return that Symbol as the label for this tensor and don't change the
        # number of nodes or add anything to the converted plan.
        return tree, last_node_label, plan

    else
        # First, get the label for the tensor created by contracting the tree
        # in plan[1] and append its contraction plan to converted plan. Then
        # do the same for plan[2]. Finally, append the contraction of these
        # trees to the converted plan.
        a, last_node_label = convert_tree_to_plan(tree[1], last_node_label,
                                                  plan)
        b, last_node_label = convert_tree_to_plan(tree[2], last_node_label,
                                                  plan)
        append!(plan, [[a, b]])
        last_node_label = last_node_label + 1
        return Symbol("node_$last_node_label"), last_node_label, plan
    end
end

"""
    function KaHyPar_contraction!(network::TensorNetworkCircuit,
                                  k::Int, ϵ::Real, V::Int,)

Function to contract a tensor network according to a contraction plan found
using the KaHyPar method.
"""
function KaHyPar_contraction!(network::TensorNetworkCircuit,
                              k::Int=2, ϵ::Real=1, V::Int=7,
                              output_shape::Union{String, Array{<:Integer, 1}}="")
    contraction_tree = KaHyPar_contraction_plan(network, k, ϵ, V)
    contraction_plan = convert_tree_to_plan(contraction_tree,
                                            network.counters["node"])

    contract_network!(network, contraction_plan[3], output_shape)
end
