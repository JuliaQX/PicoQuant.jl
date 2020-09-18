using PyCall
using JSON
using DataStructures
using LinearAlgebra

export TensorNetworkCircuit
export Node, Edge, add_gate!, edges
export new_label!, add_input!, add_output!
export load_qasm_as_circuit_from_file, load_qasm_as_circuit
export convert_qiskit_circ_to_network, transpile_circuit
export to_dict, to_json, network_from_dict, edge_from_dict, node_from_dict
export network_from_json
export inneighbours, outneighbours, virtualneighbours, neighbours
export inedges, outedges, virtualedges
export decompose_gate!
export gate_tensor

# *************************************************************************** #
#           Tensor network circuit data structure and functions
# *************************************************************************** #

"Struct to represent a node in tensor network graph"
struct Node
    # the indices that the node contains
    indices::Array{Symbol, 1}
    # tensor dimensions
    dims::Array{<:Integer, 1}
    # The symbol used to identify the tensor associated with this node.
    data_label::Symbol
end

"""
    function Node(data_label::Symbol)

Outer constructor to create an instance of Node with the given data label and no
index labels
"""
function Node(data_label::Symbol)
    Node(Array{Symbol, 1}(), Array{Int64, 1}(), data_label)
end

"""Struct to represent an edge"""
mutable struct Edge
    src::Union{Symbol, Nothing}
    dst::Union{Symbol, Nothing}
    qubit::Union{Int64, Nothing}
    virtual::Bool
    Edge(a::Union{Symbol, Nothing},
         b::Union{Symbol, Nothing},
         qubit::Union{Integer, Nothing},
         virtual::Bool=false) = new(a, b, qubit, virtual)
    Edge() = new(nothing, nothing, nothing, false)
end

"""Struct for tensor network graph of a circuit"""
struct TensorNetworkCircuit
    # Backend used to store tensors
    #FIXME: This should be parameterised, along with the TensorNetworkCircuit
    backend::AbstractBackend

    number_qubits::Integer

    # Reference to indices connecting to input qubits
    input_qubits::Array{Symbol, 1}

    # Reference to indices connecting to output qubits
    output_qubits::Array{Symbol, 1}

    # Map of nodes by symbol
    nodes::OrderedDict{Symbol, Node}

    # Dictionary of edges indexed by index symbol where value is node pair
    edges::OrderedDict{Symbol, Edge}

    # Array with indices of quantum register positions corresponding to
    # each classical register position
    qubit_ordering::Array{Integer, 1}

    # implementation details, not shared outside module
    # counters for assigning unique symbol names to nodes and indices
    counters::Dict{String, Integer}

    # maps nodes to layer numbers
    node_layers::Dict{Symbol, Integer}
end

"""
    function TensorNetworkCircuit(qubits::Int, backend::AbstractBackend=InteractiveBackend())

Outer constructor to create an instance of TensorNetworkCircuit for an empty
circuit with the given number of qubits.
"""
function TensorNetworkCircuit(qubits::Integer, backend::AbstractBackend=InteractiveBackend())
    # Create labels for edges of the network connecting input to output qubits
    index_labels = [Symbol("index_", i) for i in 1:qubits]

    # Create dictionary map from index label to node positions
    edges = OrderedDict{Symbol, Edge}()
    for i in 1:qubits
        edges[index_labels[i]] = Edge(nothing, nothing, i, false)
    end

    input_indices = index_labels
    output_indices = copy(index_labels)

    nodes = OrderedDict{Symbol, Node}()

    counters = Dict{String, Integer}()
    counters["index"] = qubits
    counters["node"] = 0
    counters["layer"] = 0

    node_layers = Dict{Symbol, Int64}()

    # Create the tensor network
    TensorNetworkCircuit(backend, qubits, input_indices, output_indices, nodes,
                         edges, collect(1:qubits), counters, node_layers)
end

# *************************************************************************** #
#                  TensorNetworkCircuit backend functions
# *************************************************************************** #

for func in [:save_tensor_data , :load_tensor_data, :contract_tensors, :save_output,
             :reshape_tensor, :permute_tensor, :decompose_tensor!, :delete_tensor!,
             :view_tensor!]
    @eval begin
        function $func(network::TensorNetworkCircuit, args...; kwargs...)
            $func(network.backend, args...; kwargs...)
        end
    end
end

# *************************************************************************** #
#                  TensorNetworkCircuit mutation functions
# *************************************************************************** #

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
                       targetqubits::Array{Integer, 1};
                       decompose::Bool=false)

Add a node to the tensor network for the given gate acting on the given quibits
"""
function add_gate!(network::TensorNetworkCircuit,
                   gate_data::Array{<:Number},
                   target_qubits::Array{<:Integer,1};
                   decompose::Bool=false)

    n = length(target_qubits)
    # create new indices for connecting gate to outputs
    input_indices = [network.output_qubits[i] for i in target_qubits]
    output_indices = [new_label!(network, "index") for _ in 1:n]

    # remap output qubits
    for (i, target_qubit) in enumerate(target_qubits)
        network.output_qubits[target_qubit] = output_indices[i]
    end

    layer = network.counters["layer"] += 1
    if decompose && n == 2
        gates_data = decompose_gate!(gate_data)
        virtual_index = new_label!(network, "index")
        node_labels = [new_label!(network, "node") for x in 1:2]
        network.edges[virtual_index] = Edge(node_labels[1],
                                            node_labels[2],
                                            nothing,
                                            true)
        for i in 1:2
            node_label = node_labels[i]
            data_label = node_label
            if i == 1
                indices = [input_indices[i], output_indices[i], virtual_index]
            else
                indices = [virtual_index, input_indices[i], output_indices[i]]
            end
            new_node = Node(indices, [size(gates_data[i])...], data_label)
            network.nodes[node_label] = new_node
            network.node_layers[node_label] = layer

            # Save the gate data to the executer
            save_tensor_data(network, node_label, data_label, gates_data[i])

            input_index = input_indices[i]
            output_index = output_indices[i]
            network.edges[output_index] = Edge(node_label,
                                               network.edges[input_index].dst,
                                               target_qubits[i])

            # if there is an output node, we need to update incoming index
            if network.edges[input_index].dst !== nothing
                out_node = network.nodes[network.edges[input_index].dst]
                for i in 1:length(out_node.indices)
                    if out_node.indices[i] == input_index
                        out_node.indices[i] = output_index
                    end
                end
            end
            network.edges[input_index].dst = node_label
        end
        return node_labels
    else
        # create a node object for the gate
        # TODO: Having data_label equal the node_label is redundant
        # but thinking these can differ when avoiding duplication of tensor data.
        node_label = new_label!(network, "node")
        data_label = node_label
        new_node = Node(vcat(input_indices, output_indices), [size(gate_data)...], data_label)
        network.nodes[node_label] = new_node
        network.node_layers[node_label] = layer

        # Save the gate data to the executer
        save_tensor_data(network, node_label, data_label, gate_data)

        # remap nodes that edges are connected to
        for qubit in 1:length(input_indices)
            input_index = input_indices[qubit]
            output_index = output_indices[qubit]
            network.edges[output_index] = Edge(node_label,
                                               network.edges[input_index].dst,
                                               target_qubits[qubit])

            # if there is an output node, we need to update incoming index
            if network.edges[input_index].dst !== nothing
                out_node = network.nodes[network.edges[input_index].dst]
                for i in 1:length(out_node.indices)
                    if out_node.indices[i] == input_index
                        out_node.indices[i] = output_index
                    end
                end
            end
            network.edges[input_index].dst = node_label
        end
        return node_label
    end
end

"""
    function edges(network::TensorNetworkCircuit)

Function to return edges from a given network
"""
function edges(network::TensorNetworkCircuit)
    network.edges
end

"""
    function add_input!(network::TensorNetworkCircuit, config::String)

Function to add input nodes to a tensor network circuit with a given input
configuration
"""
function add_input!(network::TensorNetworkCircuit, config::String)
    @assert length(config) == network.number_qubits
    for (input_index, config_char) in zip(network.input_qubits, config)

        # add guard to check if node does not already exist
        if network.edges[input_index].src === nothing
            node_label = new_label!(network, "node")
            data_label = node_label

            node_map = Dict('0' => [1., 0],
                            '1' => [0., 1.],
                            '+' => 1/sqrt(2)*[1., 1.],
                            '-' => 1/sqrt(2)*[1., -1.])

            node_data = node_map[config_char]
            dims = [size(node_data)...]
            network.nodes[node_label] = Node([input_index], dims, data_label)
            network.node_layers[node_label] = 0

            # Save the gate data to the executer
            save_tensor_data(network, node_label, data_label, node_data)

            network.edges[input_index].src = node_label
        else
            @warn "Input node already exists"
        end
    end
end

"""
    function add_output!(network::TensorNetworkCircuit, config::String)

Function to add output nodes to a tensor network circuit with a given output
configuration
"""
function add_output!(network::TensorNetworkCircuit, config::String)
    @assert length(config) == network.number_qubits
    for i in 1:network.number_qubits
        qubit_pos = network.qubit_ordering[i]
        output_index, config_char = network.output_qubits[qubit_pos], config[i]
        # add guard to check if node does not already exist
        if network.edges[output_index].dst === nothing
            node_label = new_label!(network, "node")
            data_label = node_label

            node_data = (config_char == '0') ? [1., 0.] : [0., 1.]
            dims = [size(node_data)...]
            network.nodes[node_label] = Node([output_index], dims, data_label)
            network.node_layers[node_label] = -1

            # Save the gate data to the executer
            save_tensor_data(network, node_label, data_label, node_data)

            network.edges[output_index].dst = node_label
        end
    end
end

"""
    function inneighbours(network::TensorNetworkCircuit,
                          node_label::Symbol)

Function to return the nodes which are connected to the given node with
incoming edges
"""
function inneighbours(network::TensorNetworkCircuit,
                      node_label::Symbol)
    node = network.nodes[node_label]
    myarray = Array{Symbol, 1}()
    for index in node.indices
        edge = network.edges[index]
        if edge.src !== nothing && edge.src != node_label && !edge.virtual
            push!(myarray, edge.src)
        end
    end
    myarray
end

"""
    function outneighbours(network::TensorNetworkCircuit,
                           node_label::Symbol)

Function to return the nodes which are connected to the given node with
outgoing edges
"""
function outneighbours(network::TensorNetworkCircuit,
                       node_label::Symbol)
    node = network.nodes[node_label]
    myarray = Array{Symbol, 1}()
    for index in node.indices
        edge = network.edges[index]
        if edge.dst !== nothing && edge.dst != node_label && !edge.virtual
            push!(myarray, edge.dst)
        end
    end
    myarray
end

"""
    function virtualneighbours(network::TensorNetworkCircuit,
                           node_label::Symbol)

Function to return the nodes which are connected to the given node with
virtual edges
"""
function virtualneighbours(network::TensorNetworkCircuit,
                           node_label::Symbol)
    node = network.nodes[node_label]
    myarray = Array{Symbol, 1}()
    for index in node.indices
        edge = network.edges[index]
        if edge.dst !== nothing && edge.dst != node_label && edge.virtual
            push!(myarray, edge.dst)
        elseif edge.src !== nothing && edge.src != node_label && edge.virtual
            push!(myarray, edge.src)
        end
    end
    myarray
end

"""
    function neighbours(network::TensorNetworkCircuit,

Function to get all neighbouring nodes of the given node (incoming, outgoing and
virtual)
"""
function neighbours(network::TensorNetworkCircuit,
                    node_label::Symbol)
    vcat(inneighbours(network, node_label),
         outneighbours(network, node_label),
         virtualneighbours(network, node_label))
end

"""
    function inedges(network::TensorNetworkCircuit,
                     node_label::Symbol)

Function to get all incoming edges to the current node
"""
function inedges(network::TensorNetworkCircuit,
                 node_label::Symbol)
    idxs = network.nodes[node_label].indices
    [x for x in idxs if !network.edges[x].virtual && network.edges[x].dst == node_label]
end

"""
    function outedges(network::TensorNetworkCircuit,
                      node_label::Symbol)

Function to get all outgoing edges to the current node
"""
function outedges(network::TensorNetworkCircuit,
                  node_label::Symbol)
    idxs = network.nodes[node_label].indices
    [x for x in idxs if !network.edges[x].virtual && network.edges[x].src == node_label]
end

"""
    function virtualedges(network::TensorNetworkCircuit,
                          node_label::Symbol)

Function to get all virtual edges connected to the current node
"""
function virtualedges(network::TensorNetworkCircuit,
                  node_label::Symbol)
    idxs = network.nodes[node_label].indices
    [x for x in idxs if network.edges[x].virtual]
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
    function transpile_circuit(circ,
                               couplings::Union{Nothing,
                               Array{<:Array{<:Integer, 1}, 1})} = nothing))

Transpile circuit so only neighbouring qubits have gates applied to them
"""
function transpile_circuit(circ,
                           couplings::Union{Nothing,
                           Array{<: Array{<:Integer, 1}, 1}} = nothing)
    n_qubits = convert(Int, circ.n_qubits)
    transpiler = pyimport("qiskit.transpiler")
    passes = pyimport("qiskit.transpiler.passes")

    # create a copy of the circuit with measurements removed to prevent
    # duplicate measurements
    circ = circ.remove_final_measurements(inplace=false)
    # now add measurments to all output qubits
    circ.measure_all()

    if couplings == nothing
        couplings = [[i-1, i] for i = 1:n_qubits]
    end
    coupling_map = transpiler.CouplingMap(
                    PyCall.array2py([PyCall.array2py(x) for x in couplings]))

    pass = passes.BasicSwap(coupling_map=coupling_map)
    pass_manager = transpiler.PassManager(pass)
    circ = pass_manager.run(circ)

    # now look at measurements and find out if any qubits have been remapped
    ngates = length(circ.data)
    qubit_ordering = zeros(Int, n_qubits)
    for i in (ngates-n_qubits):ngates-1
        measurement = get(circ, i)
        qbit = measurement[2][1].index
        cbit = measurement[3][1].index
        qubit_ordering[cbit+1] = qbit + 1
    end
    circ.remove_final_measurements()
    circ, qubit_ordering
end

"""
    function convert_qiskit_circ_to_network(circ, backend::AbstractBackend=InteractiveBackend();
                                            decompose::Bool=false,
                                            transpile::Bool=false)

Given a qiskit circuit object, this function will convert this to a tensor
network circuit with a backend set to `backend`. If the decompose option is
true it will decompose two qubit gates to two tensors acting on each qubit
with a virtual bond connecting them. The transpile option will transpile the
circuit using the basicswap pass from qiskit to ensure that two qubit gates
are only applied between neighbouring qubits.
"""
function convert_qiskit_circ_to_network(circ, backend::AbstractBackend=InteractiveBackend();
                                        decompose::Bool=false,
                                        transpile::Bool=false)
    barrier = pyimport("qiskit.extensions.standard.barrier")
    qi = pyimport("qiskit.quantum_info")
    n_qubits = convert(Int, circ.n_qubits)
    if transpile
        circ, qubit_ordering = transpile_circuit(circ)
    else
        qubit_ordering = collect(1:n_qubits)
    end

    tng = TensorNetworkCircuit(n_qubits, backend)
    tng.qubit_ordering[:] = qubit_ordering[:]
    for gate in circ.data
        # If the gate is a barrier then skip it
        if ! pybuiltin(:isinstance)(gate[1], barrier.Barrier)
            # Need to add 1 to index when converting from python
            target_qubits = [target.index+1 for target in gate[2]]
            dims = [2 for i = 1:2*length(target_qubits)]
            data = permutedims(qi.Operator(gate[1]).data, (2, 1))
            data = reshape(data, dims...)
            add_gate!(tng, data, target_qubits, decompose=decompose)
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
    Dict("src" => edge.src, "dst" => edge.dst,
         "virtual" => edge.virtual, "qubit" => edge.qubit)
end

"""
    function edge_from_dict(dict::Dict)

Function to create an edge instance from a dictionary
"""
function edge_from_dict(d::AbstractDict)
    Edge((d["src"] === nothing) ? nothing : Symbol(d["src"]),
         (d["dst"] === nothing) ? nothing : Symbol(d["dst"]),
         d["qubit"],
         d["virtual"])
end

"""
    function to_dict(node::Node)

Function to serialise node instance to json format
"""
function to_dict(node::Node)
    node_dict = Dict{String, Any}("indices" => [String(x) for x in node.indices])
    node_dict["dims"] = string.(node.dims)
    node_dict["data_label"] = String(node.data_label)
    node_dict
end

"""
    function node_from_dict(d::Dict)

Function to create a node instance from a json string
"""
function node_from_dict(d::AbstractDict)
    indices = [Symbol(x) for x in d["indices"]]
    dims = parse.(Int, d["dims"])
    data_label = Symbol(d["data_label"])
    Node(indices, dims, data_label)
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
    top_level["qubit_ordering"] = [x for x in network.qubit_ordering]
    top_level["node_layers"] = OrderedDict(String(k) => v
                                           for (k, v) in network.node_layers)
    top_level
end

"""
    function network_from_dict(dict::Dict{String, Any}, backend::AbstractBackend=InteractiveBackend())

Convert a dictionary to a tensor network with a given backend (default: InteractiveBackend)
"""
function network_from_dict(dict::AbstractDict{String, Any}, backend::AbstractBackend=InteractiveBackend())
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
    qubit_ordering = dict["qubit_ordering"]
    node_layers = Dict(Symbol(k) => v for (k, v) in dict["node_layers"])

    TensorNetworkCircuit(backend, number_qubits, input_qubits, output_qubits, nodes,
                         edges, qubit_ordering, counters, node_layers)
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
    function network_from_json(json_str::String, backend::AbstractBackend=InteractiveBackend())

Convert a json string to a tensor network
"""
function network_from_json(json_str::String, backend::AbstractBackend=InteractiveBackend())
    dict = JSON.parse(json_str, dicttype=OrderedDict)
    network_from_dict(dict, backend)
end

"""
    function decompose_gate!(gate_data::Array{<:Number, 4},
                             threshold::AbstractFloat=1e-15)

Decompose a tensor into two smaller tensors
"""
function decompose_gate!(gate_data::Array{<:Number, 4},
                         threshold::AbstractFloat=1e-15)
    left_positions = [1, 3]
    right_positions = [2, 4]
    dims = size(gate_data)
    left_dims = [dims[x] for x in left_positions]
    right_dims = [dims[x] for x in right_positions]

    A = permutedims(gate_data, vcat(left_positions, right_positions))
    A = reshape(A, Tuple([prod(left_dims), prod(right_dims)]))

    # Use SVD here but QR could also be used
    F = svd(A)

    # find number of singular values above the threshold
    chi = sum(F.S .> threshold)
    s = sqrt.(F.S[1:chi])

    # assume that singular values and basis of U and V matrices are sorted
    # in descending order of singular value
    B = reshape(F.U[:, 1:chi] * Diagonal(s), Tuple(vcat(left_dims, [chi,])))
    C = reshape(Diagonal(s) * F.Vt[1:chi, :], Tuple(vcat([chi,], right_dims)))

    return B, C
end

# *************************************************************************** #
#                            User utility functions
# *************************************************************************** #

# A dictionary of commonly used quantum gates.
GATE_TENSORS = Dict{Symbol, Any}()

GATE_TENSORS[:I] = [1 0; 0 1]
GATE_TENSORS[:X] = [0 1; 1 0]
GATE_TENSORS[:Y] = [0 -1im; 1im 0]
GATE_TENSORS[:Z] = [1 0; 0 -1]

GATE_TENSORS[:H] = [1 1; 1 -1]/sqrt(2)
GATE_TENSORS[:S] = [1 0; 0 1im]
GATE_TENSORS[:T] = [1 0; 0 (1 + 1im)/sqrt(2)]

GATE_TENSORS[:CX] = reshape([1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0], 2, 2, 2, 2)
GATE_TENSORS[:CZ] = reshape([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1], 2, 2, 2, 2)
GATE_TENSORS[:SWAP] = reshape([1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1], 2, 2, 2, 2)

# Toffoli gate
TOFF = zeros(8,8); TOFF[7,8] = 1; TOFF[8,7] = 1
for i = 1:6
    TOFF[i,i] = 1
end
GATE_TENSORS[:TOFF] = reshape(TOFF, 2, 2, 2, 2, 2, 2)

# IBM gates
GATE_TENSORS[:U3] = function (θ::Real,ϕ::Real,λ::Real)
    [cos(θ/2) -exp(1im*λ)*sin(θ/2);
     exp(1im*ϕ)*sin(θ/2) exp(1im*(ϕ+λ))*cos(θ/2)]
end

GATE_TENSORS[:U2] = function (ϕ::Real, λ::Real)
    GATE_TENSORS[:U3](0, ϕ, λ)
end

GATE_TENSORS[:U1] = function (λ::Real)
    GATE_TENSORS[:U3](0, 0, λ)
end

"""
    function gate_tensor(gate::Symbol)

Function to return tensors commonly used in quantum circuits. Input argument
should be one of the following symbols:
:I, :X, :Y, :Z, :H, :S, :T, :CX, :CZ, :SWAP, :TOFF, :U3, :U2, :U1
"""
function gate_tensor(gate::Symbol)

    if !(gate in keys(GATE_TENSORS))
        inputs = [":$key" for key in keys(GATE_TENSORS)]
        println("Accepted input: ", join(inputs, ", "))
        error("Invalid input to gate_tensor: $(gate)")
    end

    gate = GATE_TENSORS[gate]
    if typeof(gate) <: Array{<:Number}
        return copy(gate)
    end
    gate
end
