using PyCall
using JSON
using DataStructures

export TensorNetworkCircuit, Node, add_gate!, edges
export load_qasm_as_circuit_from_file, load_qasm_as_circuit
export convert_qiskit_circ_to_network
export to_dict, from_dict, network_to_json, network_from_json



struct Node
    indices::Array{Symbol, 1}
    data::Array{<:Number}
end

"""
    function Node(data::Array{Number})

Outer constructor to create an instance of Node with the given data and no
index labels
"""
function Node(data::Array{<:Number})
    Node(Array{Symbol, 1}(), data)
end

struct TensorNetworkCircuit
    output_qubits::Array{Node, 1}
    nodes::Array{Node, 1}
    index_map::Dict{Symbol, Array{<:Integer, 1}}
end

"""
    function TensorNetworkCircuit(qubits::Integer)

Outer constructor to create an instance of TensorNetworkCircuit for an empty
circuit with the given number of qubits.
"""
function TensorNetworkCircuit(qubits::Integer)

    # Create labels for edges of the network connecting input to output qubits
    labels = [Symbol("ind_", i) for i in 1:qubits]

    # Create Nodes for the input and output qubits
    in_qubits = [Node([labels[i]], [1.,0.]) for i in 1:qubits]
    out_qubits = [Node([labels[i]], [1.,0.]) for i in 1:qubits]

    # Create dictionary map from index label to node positions
    index_map = Dict{Symbol, Array{<:Integer, 1}}()
    for i in 1:qubits
        index_map[labels[i]] = [i, i+qubits]
    end

    # Create the tensor network
    TensorNetworkCircuit(out_qubits, [in_qubits; out_qubits], index_map)
end



"""
    function add_gate!(network::TensorNetworkCircuit,
                       gate::Node,
                       targetqubits::Array{Integer, 1})

Add a node to the tensor network for the given gate acting on the given quibits
"""
function add_gate!(network::TensorNetworkCircuit,
                   gate::Node,
                   target_qubits::Array{<:Integer,1})

    # Create an array of qubits the gate acts on (array of Nodes)
    targets = [network.output_qubits[i] for i in target_qubits]

    # Create index labels for the new edges connecting the gate to the output
    # qubits (labels) and an array of index labels for the gate itself (indices)
    ind_num = length(keys(network.index_map))
    labels = [Symbol("ind_", i) for i in (ind_num+1):(ind_num+length(targets))]
    old_indices = [target.indices for target in targets]
    old_indices = reduce(vcat, old_indices)
    indices = vcat(old_indices, labels)

    # Set the new index labels for the new gate
    append!(gate.indices, indices)

    # Relabel the target qubits so that they're contracted with the new gate
    for i in 1:length(targets)
        targets[i].indices[1] = labels[i]
    end

    # Add the gate to the list of nodes
    append!(network.nodes, [gate])

    # Update the index_map dictionary with the new labels and gate
    for (old_index, new_index, target) in zip(old_indices, labels, targets)

        # Get the positions of the tensor being relabeled and the new gate
        tensor_pair = network.index_map[old_index]
        target_position = tensor_pair[2]
        gate_position = length(network.nodes)

        # Replace the position index of the old target with that of the new gate
        tensor_pair[2] = gate_position

        # Append the positions of the tensors with the new label to index_map
        network.index_map[new_index] = [gate_position, target_position]
    end
end



"""
    function edges(network::TensorNetworkCircuit)

Return an array of pairs of integers representing the edges of the tensor
network. The integers are the position indices of the nodes connected by an edge
in network.nodes
"""
function edges(network::TensorNetworkCircuit)
    # An empty array to hold the edges
    edge_list = Array{Array{Int, 1}, 1}()

    # Loop over all index labels
    for edge_label in keys(network.index_map)
        # Get the pair of nodes which have the index label "edge_label"
        edge = network.index_map[edge_label]

        # If the pair isn't already in the array then append it.
        # This avoids multiply appending node pairs which have more than one
        # edge connecting them
        if ! (edge in edge_list)
            append!(edge_list, [edge])
        end
    end
    return edge_list
end



"""
    function load_qasm_as_circuit_from_file(qasm_path::String)

Function to load qasm from the given path and return a qiskit circuit
"""
function load_qasm_as_circuit_from_file(qasm_path::String)
    qiskit = pyimport("qiskit")
    if isfile(qasm_path)
        return qiskit.QuantumCircuit.from_qasm_file(qasm_path)
    else
        return false
    end
end

"""
    function load_qasm_as_circuit(qasm_str::String)

Function to load qasm from a given qasm string and return a qiskit circuit
"""
function load_qasm_as_circuit(qasm_str::String)
    qiskit = pyimport("qiskit")
    return qiskit.QuantumCircuit.from_qasm_str(qasm_str)
end



"""
    function convert_qiskit_circ_to_network(circ)

Convert the given a qiskit circuit to a tensor network
"""
function convert_qiskit_circ_to_network(circ)
    transpiler = pyimport("qiskit.transpiler")
    passes = pyimport("qiskit.transpiler.passes")
    qi = pyimport("qiskit.quantum_info")

    coupling = [[i-1, i] for i = 1:circ.n_qubits]
    coupling_map = transpiler.CouplingMap(
                   PyCall.array2py([PyCall.array2py(x) for x in coupling]))

    pass = passes.BasicSwap(coupling_map=coupling_map)
    # pass = passes.LookaheadSwap(coupling_map=coupling_map)
    # pass = passes.StochasticSwap(coupling_map=coupling_map)
    pass_manager = transpiler.PassManager(pass)
    transpiled_circ = pass_manager.run(circ)

    tng = TensorNetworkCircuit(transpiled_circ.n_qubits)
    for gate in transpiled_circ.data
        # Need to add 1 to index when converting from python to julia indexing
        target_qubits = [target.index+1 for target in gate[2]]
        dims = [2 for i = 1:2*length(target_qubits)]
        data = reshape(qi.Operator(gate[1]).data, dims...)
        gate_node = Node(Array{Symbol, 1}(), data)
        add_gate!(tng, gate_node, target_qubits)
    end

    tng
end



"""
    function to_dict(network::TensorNetworkCircuit)

Convert a tensor network to a nested dictionary
"""
function to_dict(network::TensorNetworkCircuit)
    top_level = OrderedDict()
    top_level["num_qubits"] = length(network.output_qubits)
    top_level["index_map"] = Dict(String(k)=>v
                                for (k,v) in pairs(network.index_map))

    top_level["nodes"] = OrderedDict{Int64, Any}()
    nodes_dict = top_level["nodes"]
    for (n,node) in enumerate(network.nodes)
        nodes_dict[n] = Dict{Symbol, Any}(:labels=>node.indices)
        nodes_dict[n][:dims] = size(node.data)
        nodes_dict[n][:data_re] = reshape(real.(node.data), length(node.data))
        nodes_dict[n][:data_im] = reshape(imag.(node.data), length(node.data))
    end

    top_level
end

"""
    function from_dict(top_level::Dict{String, Any})

Convert a dictionary to a tensor network
"""
function from_dict(top_level::Dict{String, Any})
    # Get the index map and convert it to the right type
    index_map = Dict(Symbol(k) => Array{Int, 1}(v)
                     for (k,v) in pairs(top_level["index_map"]))

    nodes_dict = top_level["nodes"]
    nodes = Array{Node, 1}(undef, length(nodes_dict))

    for i = 1:length(nodes_dict)
        indices = Symbol.(nodes_dict["$i"]["labels"])
        data = nodes_dict["$i"]["data_re"] .+ nodes_dict["$i"]["data_im"].*1im
        data = reshape(data, nodes_dict["$i"]["dims"]...)
        nodes[i] = Node(indices, data)
    end

    num_qubits = top_level["num_qubits"]
    output = [nodes[i] for i in num_qubits+1:2*num_qubits]

    TensorNetworkCircuit(output, nodes, index_map)
end



"""
    function network_to_json(tng::TensorNetworkCircuit)

Convert a tensor network to a json string
"""
function network_to_json(tng::TensorNetworkCircuit, indent::Integer=0)
    dict = to_dict(tng)
    if indent == 0
        return JSON.json(dict)
    else
        return JSON.json(dict, indent)
    end
end

"""
    function network_from_json(json_str::String)

Convert a json string to a tensor network
"""
function network_from_json(json_str::String)
    dict = JSON.parse(json_str)
    from_dict(dict)
end
