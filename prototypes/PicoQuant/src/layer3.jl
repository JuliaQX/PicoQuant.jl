using PyCall
using JSON
using DataStructures

export TensorNetworkCircuit
export Node, Edge, add_gate!, edges
export new_label!, add_input!, add_output!
export load_qasm_as_circuit_from_file, load_qasm_as_circuit
export convert_qiskit_circ_to_network
export to_dict, to_json, network_from_dict, edge_from_dict, node_from_dict
export network_from_json

# *************************************************************************** #
#           Tensor network circuit data structure and functions
# *************************************************************************** #

"Struct to represent a node in tensor network graph"
struct Node
    # the indices that the node contains
    indices::Array{Symbol, 1}
    # data tensor
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

"Struct to represent an edge"
mutable struct Edge
    src::Union{Symbol, Nothing}
    dst::Union{Symbol, Nothing}
    Edge(a::Union{Symbol, Nothing}, b::Union{Symbol, Nothing}) = new(a, b)
    Edge() = new(nothing, nothing)
end

"Struct for tensor network graph of a circuit"
struct TensorNetworkCircuit
    number_qubits::Integer

    # Reference to indices connecting to input qubits
    input_qubits::Array{Symbol, 1}

    # Reference to indices connecting to output qubits
    output_qubits::Array{Symbol, 1}

    # Map of nodes by symbol
    nodes::OrderedDict{Symbol, Node}

    # dictionary of edges indexed by index symbol where value is node pair
    edges::OrderedDict{Symbol, Edge}

    # implementation details, not shared outside module
    # counters for assigning unique symbol names to nodes and indices
    counters::Dict{String, Integer}
end

"""
    function TensorNetworkCircuit(qubits::Integer)

Outer constructor to create an instance of TensorNetworkCircuit for an empty
circuit with the given number of qubits.
"""
function TensorNetworkCircuit(qubits::Integer)
    # Create labels for edges of the network connecting input to output qubits
    index_labels = [Symbol("index_", i) for i in 1:qubits]

    # Create dictionary map from index label to node positions
    edges = OrderedDict{Symbol, Edge}()
    for i in 1:qubits
        edges[index_labels[i]] = Edge()
    end

    input_indices = index_labels
    output_indices = copy(index_labels)

    nodes = OrderedDict{Symbol, Node}()

    counters = Dict{String, Integer}()
    counters["index"] = qubits
    counters["node"] = 0

    # Create the tensor network
    TensorNetworkCircuit(qubits, input_indices, output_indices, nodes,
                         edges, counters)
end

"""
    function new_label!(network::TensorNetworkCircuit, label_str)

Function to create a unique symbol by incrememting relevant counter
"""
function new_label!(network::TensorNetworkCircuit, label_str)
    network.counters[label_str] += 1
    Symbol(label_str, "_", network.counters[label_str])
end

"""
    function add_gate!(network::TensorNetworkCircuit,
                       gate_data::Array{<:Number},
                       targetqubits::Array{Integer, 1})

Add a node to the tensor network for the given gate acting on the given quibits
"""
function add_gate!(network::TensorNetworkCircuit,
                   gate_data::Array{<:Number},
                   target_qubits::Array{<:Integer,1})

    n = length(target_qubits)
    # create new indices for connecting gate to outputs
    input_indices = [network.output_qubits[i] for i in target_qubits]
    output_indices = [new_label!(network, "index") for _ in 1:n]

    # remap output qubits
    for (i, target_qubit) in enumerate(target_qubits)
        network.output_qubits[target_qubit] = output_indices[i]
    end

    # create a node object for the gate
    node_label = new_label!(network, "node")
    new_node = Node(vcat(input_indices, output_indices), gate_data)
    network.nodes[node_label] = new_node

    # remap nodes that edges are connected to
    for (input_index, output_index) in zip(input_indices, output_indices)
        network.edges[output_index] = Edge(node_label,
                                           network.edges[input_index].dst)
        network.edges[input_index].dst = node_label
    end
end

function edges(network::TensorNetworkCircuit)
    network.edges
end

function add_input!(network::TensorNetworkCircuit, config::String)
    @assert length(config) == network.number_qubits
    for (input_index, config_char) in zip(network.input_qubits, config)
        node_label = new_label!(network, "node")
        node_data = (config_char == '0') ? [1., 0.] : [0., 1.]
        network.nodes[node_label] = Node([input_index], node_data)
        network.edges[input_index].src = node_label
    end
end

function add_output!(network::TensorNetworkCircuit, config::String)
    @assert length(config) == network.number_qubits
    for (output_index, config_char) in zip(network.output_qubits, config)
        node_label = new_label!(network, "node")
        node_data = (config_char == '0') ? [1., 0.] : [0., 1.]
        network.nodes[node_label] = Node([output_index], node_data)
        network.edges[output_index].dst = node_label
    end
end

# *************************************************************************** #
#                  Functions to read circuit from qasm
# *************************************************************************** #

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
    barrier = pyimport("qiskit.extensions.standard.barrier")

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
        # If the gate is a barrier then skip it
        if ! pybuiltin(:isinstance)(gate[1], barrier.Barrier)
            # Need to add 1 to index when converting from python
            target_qubits = [target.index+1 for target in gate[2]]
            dims = [2 for i = 1:2*length(target_qubits)]
            data = reshape(qi.Operator(gate[1]).data, dims...)
            add_gate!(tng, data, target_qubits)
        end
    end

    tng
end

# *************************************************************************** #
#                    To/from functions for TN circuits
# *************************************************************************** #

"""
    function to_dict(edge::Edge)

Function to convert an edge instance to a serialisable dictionary
"""
function to_dict(edge::Edge)
    Dict("src" => edge.src, "dst" => edge.dst)
end

"""
    function edge_from_dict(dict::Dict)

Function to create an edge instance from a dictionary
"""
function edge_from_dict(d::AbstractDict)
    Edge((d["src"] == nothing) ? nothing : Symbol(d["src"]),
         (d["dst"] == nothing) ? nothing : Symbol(d["dst"]))
end

"""
    function to_dict(node::Node)

Function to serialise node instance to json format
"""
function to_dict(node::Node)
    node_dict = Dict{String, Any}("indices" => [String(x) for x in node.indices])
    if ndims(node.data) == 0
        node_dict["data_re"] = real(node.data)
        node_dict["data_im"] = imag(node.data)
        node_dict["data_dims"] = (1, )
    else
        node_dict["data_re"] = reshape(real.(node.data),
                                          length(node.data))
        node_dict["data_im"] = reshape(imag.(node.data),
                                          length(node.data))
        node_dict["data_dims"] = size(node.data)
    end
    node_dict
end

"""
    function node_from_dict(d::Dict)

Function to create a node instance from a json string
"""
function node_from_dict(d::AbstractDict)
    indices = [Symbol(x) for x in d["indices"]]
    data = reshape(d["data_re"] + d["data_im"].*1im, Tuple(d["data_dims"]))
    Node(indices, data)
end

"""
    function to_dict(network::TensorNetworkCircuit)

Convert a tensor network to a nested dictionary
"""
function to_dict(network::TensorNetworkCircuit)
    top_level = OrderedDict{String, Any}()
    top_level["number_qubits"] = network.number_qubits
    top_level["edges"] = OrderedDict(String(k) => to_dict(v)
                                for (k,v) in pairs(network.edges))

    top_level["nodes"] = OrderedDict{String, Any}()
    nodes_dict = top_level["nodes"]
    for (node_label, node) in pairs(network.nodes)
        nodes_dict[String(node_label)] = to_dict(node)
    end
    top_level["input_qubits"] = [String(x) for x in network.input_qubits]
    top_level["output_qubits"] = [String(x) for x in network.output_qubits]
    top_level
end

"""
    function network_from_dict(dict::Dict{String, Any})

Convert a dictionary to a tensor network
"""
function network_from_dict(dict::AbstractDict{String, Any})
    number_qubits = dict["number_qubits"]
    # initialise counters
    counters = Dict("index" => 0, "node" => 0)

    # Get the index map and convert it to the right type
    edges = OrderedDict{Symbol, Edge}()
    for (k,v) in pairs(dict["edges"])
        edge_num = parse(Int64, split(k, "_")[end])
        counters["index"] = max(counters["index"], edge_num)
        edges[Symbol(k)] = edge_from_dict(v)
    end

    nodes = OrderedDict{Symbol, Node}()
    for (k, v) in dict["nodes"]
        node_num = parse(Int64, split(k, "_")[end])
        counters["node"] = max(counters["node"], node_num)
        nodes[Symbol(k)] = node_from_dict(v)
    end

    input_qubits = [Symbol(x) for x in dict["input_qubits"]]
    output_qubits = [Symbol(x) for x in dict["output_qubits"]]

    TensorNetworkCircuit(number_qubits, input_qubits, output_qubits, nodes,
                         edges, counters)
end



"""
    function to_json(tng::TensorNetworkCircuit)

Convert a tensor network to a json string
"""
function to_json(tng::TensorNetworkCircuit, indent::Integer=0)
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
    dict = JSON.parse(json_str, dicttype=OrderedDict)
    network_from_dict(dict)
end
