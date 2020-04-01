using PyCall
using LightGraphs
using MetaGraphs
using DataStructures
using JSON

export load_qasm_as_circuit, load_qasm_as_circuit_from_file, convert_to_tensor_network_graph
export TensorNetworkCircuit, to_json, tng_from_json

struct TensorNetworkCircuit
    qubits::Integer
    graph::MetaDiGraph
    inputs::Array{Integer, 1}
    outputs::Array{Integer, 1}
end

"""
    function TensorNetworkCircuit(qubits::Integer)

Outer constructor to create an instance of TensorNetworkCircuit
"""
function TensorNetworkCircuit(qubits::Integer)
    graph = MetaDiGraph(qubits)
    # add input nodes
    inputs = collect(1:qubits)
    for i = 1:qubits
        set_props!(graph, i, Dict(:qubits => [i],
                                  :indices => [1],
                                  :dims => [2],
                                  :data => [1., 0.],
                                  :type => "input",
                                  :data_order => "col"))
    end

    # add output nodes
    outputs = collect((qubits + 1):(qubits * 2))
    for i = 1:qubits
        add_vertex!(graph)
        set_props!(graph, i + qubits, Dict(:qubits => [i + qubits],
                                  :indices => [1],
                                  :dims => [2],
                                  :data => [1., 0.],
                                  :type => "output",
                                  :data_order => "col"))
        # add edge from input to output
        add_edge!(graph, i, i + qubits)
        set_prop!(graph, i, i + qubits, :indices, [1, 1])
    end
    TensorNetworkCircuit(qubits, graph, inputs, outputs)
end

"""
    function add_gate!(tng::TensorNetworkCircuit, gate::Array{ComplexF64, 2}, qubits::Array{Integer, 1})

Add gate with matrix op given to circuit
"""
function add_gate!(tng::TensorNetworkCircuit, gate::Array{ComplexF64, 2}, qubits::Array{T, 1}) where T <: Integer
    # add a vertex for this gate
    add_vertex!(tng.graph)
    vertex_id = length(vertices(tng.graph))

    # prepare property values
    props = Dict{Symbol, Any}()
    props[:qubits] = qubits
    props[:indices] = collect(1:length(qubits)*2)
    props[:dims] = collect([2 for i = 1:length(qubits)*2])
    props[:data] = reshape(gate, prod(size(gate)))
    props[:type] = "gate"
    props[:data_order] = "col"

    set_props!(tng.graph, vertex_id, props)

    for (i, qubit) in enumerate(qubits)
        qubit_index = qubit + 1
        output_node = tng.outputs[qubit_index]
        from_node = inneighbors(tng.graph, output_node)[1]
        edge_indices = get_prop(tng.graph, from_node, output_node, :indices)
        rem_edge!(tng.graph, from_node, output_node)

        add_edge!(tng.graph, from_node, vertex_id)
        set_prop!(tng.graph, from_node, vertex_id, :indices, [edge_indices[1], i])

        add_edge!(tng.graph, vertex_id, output_node)
        set_prop!(tng.graph, vertex_id, output_node, :indices, [length(qubits) + i, edge_indices[2]])
    end
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
    function convert_to_tensor_network_graph(circ)

Given a qiskit circuit, convert to a tensor network graph
"""
function convert_to_tensor_network_graph(circ)
    transpiler = pyimport("qiskit.transpiler")
    passes = pyimport("qiskit.transpiler.passes")

    coupling = [[i-1, i] for i = 1:circ.n_qubits]

    coupling_map = transpiler.CouplingMap(PyCall.array2py([PyCall.array2py(x) for x in coupling]))

    pass = passes.BasicSwap(coupling_map=coupling_map)
    # pass = passes.LookaheadSwap(coupling_map=coupling_map)
    # pass = passes.StochasticSwap(coupling_map=coupling_map)
    pass_manager = transpiler.PassManager(pass)
    transpiled_circ = pass_manager.run(circ)

    tng = TensorNetworkCircuit(transpiled_circ.n_qubits)
    for gate in transpiled_circ.data
        add_gate!(tng, gate[1].to_matrix(), [x.index for x in gate[2]])
    end

    tng
end

"""
    function to_dict(tng::TensorNetworkCircuit)

Convert a nested dictionary from a tensor network circuit struct
"""
function to_dict(tng::TensorNetworkCircuit)
    top_level = OrderedDict()
    top_level["inputs"] = tng.inputs
    top_level["outputs"] = tng.outputs

    top_level["nodes"] = OrderedDict{Int64, Any}()
    nodes_dict = top_level["nodes"]
    for node in vertices(tng.graph)
        nodes_dict[node] = deepcopy(props(tng.graph, node))
        data = nodes_dict[node][:data]
        nodes_dict[node][:data_re] = reshape(real.(data), prod(size(data)))
        nodes_dict[node][:data_im] = reshape(imag.(data), prod(size(data)))
        delete!(nodes_dict[node], :data)
    end

    top_level["edges"] = OrderedDict{Int64, Any}()
    edges_dict = top_level["edges"]
    for (i, edge) in enumerate(edges(tng.graph))
        edges_dict[i] = props(tng.graph, edge)
        edges_dict[i][:src] = edge.src
        edges_dict[i][:dst] = edge.dst
    end

    top_level
end

"""
    function to_json(tng::TensorNetworkCircuit)

Convert a nested dictionary from a tensor network circuit struct
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
    function from_dict(top_level::Dict{String, Any})

Convert a dictionary to a tensor network circuit graph
"""
function from_dict(top_level::Dict{String, Any})
    nodes = top_level["nodes"]

    graph = MetaDiGraph(length(nodes))
    for i = 1:length(nodes)
        node_dict = nodes["$(i)"]
        set_prop!(graph, i, :indices, convert(Array{Int64, 1}, node_dict["indices"]))
        set_prop!(graph, i, :dims, convert(Array{Int64, 1}, node_dict["dims"]))
        set_prop!(graph, i, :type, node_dict["type"])
        set_prop!(graph, i, :data_order, node_dict["data_order"])
        set_prop!(graph, i, :data, node_dict["data_re"] .+ node_dict["data_im"] .* 1im)
        set_prop!(graph, i, :qubits, convert(Array{Int64, 1}, node_dict["qubits"]))
    end

    edges = top_level["edges"]
    for i = 1:length(edges)
        edge = edges["$(i)"]
        src, dst = edge["src"], edge["dst"]
        add_edge!(graph, src, dst)
        set_prop!(graph, src, dst, :indices, convert(Array{Int64, 1}, edge["indices"]))
        set_prop!(graph, src, dst, :src, src)
        set_prop!(graph, src, dst, :dst, dst)
    end
    TensorNetworkCircuit(length(top_level["inputs"]), graph, top_level["inputs"], top_level["outputs"])
end

"""
    function tng_from_json(json_str::String)

Convert a json string to a tensor network circuit struct
"""
function tng_from_json(json_str::String)
    dict = JSON.parse(json_str)
    from_dict(dict)
end
