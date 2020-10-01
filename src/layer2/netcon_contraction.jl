import TensorOperations.optimaltree

export netcon_contraction!, netcon

# *************************************************************************** #
#                              Netcon functions
# *************************************************************************** #

"""
    function netcon(network::TensorNetworkCircuit)

This function will call the netcon implementation in tensor operations.
"""
function netcon(network::TensorNetworkCircuit, return_type::Symbol=:plan)

    # Generate the arguments for the netcon method from the given network.
    tensor_indices, index_dims, tensor_labels = create_netcon_input(network)

    # Run the netcon method to find the optimal contraction plan.
    contraction_tree, cost = optimaltree(tensor_indices, index_dims)

    if return_type == :tree
        # replace integers in contraction tree with node labels before returning
        contraction_tree = replace_leaf_integers_with_labels(contraction_tree,
                                                             tensor_labels)
        return contraction_tree

    elseif return_type == :plan
        # Convert the tree to an array of tensor pairs to contract in sequence.
        last_node_label = network.counters["node"]
        a, b, contraction_plan = convert_tree_to_plan(contraction_tree,
                                                      last_node_label,
                                                      tensor_labels)
        return contraction_plan

    else
        # If neither return option is picked, return to the user the raw netcon
        # output and labels of the tensors separately.
        return contraction_tree, tensor_labels
    end
end

"""
    function create_netcon_input(network::TensorNetworkCircuit)

This function perpares arguments for netcon to find the optimal contraction
plan of the given tensor network.
"""
function create_netcon_input(network::TensorNetworkCircuit)

    # Collect the indices and corresponding dimensions of the network
    # into arrays.
    num_tensors = length(network.nodes)

    tensor_indices = Array{Array{Symbol, 1}, 1}(undef, num_tensors)
    tensor_dims = Array{Array{<:Integer, 1}, 1}(undef, num_tensors)
    tensor_labels = Array{Symbol, 1}(undef, num_tensors)

    for (i, node) in enumerate(values(network.nodes))
        tensor_indices[i] = node.indices
        tensor_dims[i] = node.dims
        tensor_labels[i] = node.data_label
    end

    # Make a dictionary of index dimensions for netcon.
    labels = reduce(vcat, tensor_indices); dims = reduce(vcat, tensor_dims)
    index_dims = Dict{Symbol, Int}(labels .=> dims)

    # The netcon implementation only needs tensor_indices and index_dims as
    # input but it returns a nested array of integers which need to be replaced
    # by node labels for contraction. To this end, tensor_labels is also
    # returned.
    return tensor_indices, index_dims, tensor_labels
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

tensor_labels - An array of all the node labels for the nodes in a tensor
                network.

converted_plan - Initially, this argument should be an empty array. It will hold
                 the sequence of pairs when the function completes.
"""
function convert_tree_to_plan(tree::Union{Array{<:Any, 1}, Integer},
                              last_node_label::Integer,
                              tensor_labels::Array{Symbol, 1},
                              plan::Array{Array{Symbol, 1}, 1}
                                    =Array{Array{Symbol, 1}, 1}())

    # Recursively convert the plan into a sequence of pairs.
    if typeof(tree) == Int
        # Base case: If the plan is an integer corresponding to a single tensor,
        # return that integer as the label for this tensor and don't change the
        # number of nodes or add anything to the converted plan.
        return tensor_labels[tree], last_node_label, plan

    else
        # First, get the label for the tensor created by contracting the tree
        # in plan[1] and append its contraction plan to converted plan. Then
        # do the same for plan[2]. Finally, append the contraction of these
        # trees to the converted plan.
        a, last_node_label = convert_tree_to_plan(tree[1], last_node_label,
                                                  tensor_labels, plan)
        b, last_node_label = convert_tree_to_plan(tree[2], last_node_label,
                                                  tensor_labels, plan)
        append!(plan, [[a, b]])
        last_node_label = last_node_label + 1
        return Symbol("node_$last_node_label"), last_node_label, plan
    end
end

"""
    function replace_leaf_integers_with_labels(tree, labels::Array{Symbol, 1})

Function to replace the integers in a contraction tree with the corresponding
node labels.
"""
function replace_leaf_integers_with_labels(tree, labels::Array{Symbol, 1})
    # Base Case: If the given sub tree is an integer, meaning it is a leaf
    # of the parent tree, then return the corresponding node label.
    if typeof(tree) == Int
        return labels[tree]

    else
        # Replace the leaves of each branch of the given sub tree with
        # node labels.
        branch1 = replace_leaf_integers_with_labels(tree[1], labels)
        branch2 = replace_leaf_integers_with_labels(tree[2], labels)
        return [branch1, branch2]
    end
end

"""
    function netcon_contraction!(network::TensorNetworkCircuit,
                            output_shape::Union{String, Array{<:Integer, 1}}="")

Function to contraction a network according to the optimal contraction plan
found by the netcon method.
"""
function netcon_contraction!(network::TensorNetworkCircuit,
                             output_shape::Union{String, Array{<:Integer, 1}}="")

    if length(network.nodes) > 36
        error("The number of nodes in the network exceed the maximum for netcon")
    end
    contraction_plan = netcon(network)
    contract_network!(network, contraction_plan, output_shape)
end
