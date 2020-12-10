var documenterSearchIndex = {"docs":
[{"location":"layers/layer2/#Layer-2-Operations","page":"Layer 2","title":"Layer 2 Operations","text":"","category":"section"},{"location":"layers/layer2/","page":"Layer 2","title":"Layer 2","text":"Layer 2 operations are concerned with manipulating tensor network structures and are responsible for coordinating and passing off to layer 1 functions to perform the computations.","category":"page"},{"location":"layers/layer2/#Title","page":"Layer 2","title":"Title","text":"","category":"section"},{"location":"layers/layer2/","page":"Layer 2","title":"Layer 2","text":"PicoQuant.random_contraction_plan\nPicoQuant.inorder_contraction_plan\nPicoQuant.contraction_plan_to_json\nPicoQuant.contraction_plan_from_json\nPicoQuant.contract_pair!\nPicoQuant.contract_network!\nPicoQuant.full_wavefunction_contraction!\nPicoQuant.compress_tensor_chain!\nPicoQuant.contract_mps_tensor_network_circuit!\nPicoQuant.calculate_mps_amplitudes!\nPicoQuant.sort_indices\nPicoQuant.create_ncon_indices","category":"page"},{"location":"layers/layer2/#PicoQuant.random_contraction_plan","page":"Layer 2","title":"PicoQuant.random_contraction_plan","text":"random_contraction_plan(network::TensorNetworkCircuit)\n\nFunction to create a random contraction plan\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.contract_pair!","page":"Layer 2","title":"PicoQuant.contract_pair!","text":"contract_pair!(network::TensorNetworkCircuit,\n                        edge::Symbol)\n\nContract a pair of nodes connected by the given edge.\n\n\n\n\n\ncontract_pair!(network::TensorNetworkCircuit,\n                        A_label::Symbol,\n                        B_label::Symbol)\n\nFunction to contract the nodes A and B of the network.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.contract_network!","page":"Layer 2","title":"PicoQuant.contract_network!","text":"contract_network!(network::TensorNetworkCircuit,\n                           plan::Array{Symbol, 1},\n                           output_shape::Union{String, Array{Int, 1}})\n\nFunction to contract the given network according to the given contraction plan. The resulting tensor will be given the shape described by 'output_shape'.\n\n\n\n\n\ncontract_network!(network::TensorNetworkCircuit,\n                           plan::Array{Array{Symbol, 1}, 1},\n                           output_shape::Union{String, Array{Int, 1}, 1})\n\nFunction to contract the given network according to the given contraction plan. The resulting tensor will be given the shape described by 'output_shape'.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.full_wavefunction_contraction!","page":"Layer 2","title":"PicoQuant.full_wavefunction_contraction!","text":"full_wavefunction_contraction!(network::TensorNetworkCircuit,\n                                        output_shape::Union{String, Array{Int, 1}}=\"\")\n\nFunction to contract a network by first contracting input nodes together, to get the wavefunction representing the initial state, and then contracting gates into the wavefunction in the order they appear in the circuit.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.compress_tensor_chain!","page":"Layer 2","title":"PicoQuant.compress_tensor_chain!","text":"compress_tensor_chain!(network::TensorNetworkCircuit,\n                                nodes::Array{Symbol, 1};\n                                threshold::AbstractFloat=1e-13,\n                                max_rank::Int=0)\n\nCompress a chain of tensors by given by the array of symbols. This is achieved by peforming forward and backward sweeps where of compression operations on each bond. Compression of each bond proceeds by discarding singular values and corresponding axes with singular values below the given threshold or values beyond the maxrank (maxrank zero corresponds to infinite rank)\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.contract_mps_tensor_network_circuit!","page":"Layer 2","title":"PicoQuant.contract_mps_tensor_network_circuit!","text":"contract_mps_tensor_network_circuit(network::TensorNetworkCircuit;\n                                             max_bond::Int=2,\n                                             threshold::AbstractFloat=1e-13,\n                                             max_rank::Int=0)\n\nContract a tensor network representing a quantum circuit using MPS techniques\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.calculate_mps_amplitudes!","page":"Layer 2","title":"PicoQuant.calculate_mps_amplitudes!","text":"calculate_mps_amplitudes!(network::TensorNetworkCircuit,\n                                  mps_nodes::Array{Symbol, 1},\n                                  result::String=\"result\")\n\nCalculate amplitudes from an MPS state\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.sort_indices","page":"Layer 2","title":"PicoQuant.sort_indices","text":"sort_indices(A::Node, B::Node)\n\nThis function divides all of the indices of two nodes A and B into two arrays, one array for all the shared indices between A and B and a second array for all the indices that are not shared.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer2/#PicoQuant.create_ncon_indices","page":"Layer 2","title":"PicoQuant.create_ncon_indices","text":"create_ncon_indices(A::Node, B::Node,\n                             common_indices::Array{Symbol, 1},\n                             uncommon_indices::Array{Symbol, 1})\n\nThis function returns ncon indices for the nodes A and B. These can then to be passed to ncon in order to contract the tensor data associated with A and B.\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#Backend-Operations","page":"Backends","title":"Backend Operations","text":"","category":"section"},{"location":"backends/backend/","page":"Backends","title":"Backends","text":"Backends in PicoQuant aim to provide the same functionality but each is tailored for a specific purpose/hardware platform/use case. Each backend should implement the same interface. Here you will find documentation on the backend structures and the functions they implement.","category":"page"},{"location":"backends/backend/#Title","page":"Backends","title":"Title","text":"","category":"section"},{"location":"backends/backend/","page":"Backends","title":"Backends","text":"PicoQuant.AbstractBackend\nPicoQuant.DSLBackend\nPicoQuant.InteractiveBackend\nPicoQuant.save_tensor_data\nPicoQuant.load_tensor_data\nPicoQuant.save_output\nPicoQuant.reshape_tensor\nPicoQuant.decompose_tensor!\nPicoQuant.push!","category":"page"},{"location":"backends/backend/#PicoQuant.AbstractBackend","page":"Backends","title":"PicoQuant.AbstractBackend","text":"Abstract type defining backends which determine the behavior of functions that act on TensorNetworkCircuits\n\n\n\n\n\n","category":"type"},{"location":"backends/backend/#PicoQuant.DSLBackend","page":"Backends","title":"PicoQuant.DSLBackend","text":"The dsl backend will get functions that act on a tensor network circuit to write dsl commands. It also holds filenames for files containing dsl commands, tensor data and the result of fully contracting a tensor network circuit.\n\n\n\n\n\n","category":"type"},{"location":"backends/backend/#PicoQuant.InteractiveBackend","page":"Backends","title":"PicoQuant.InteractiveBackend","text":"The Interactive backend gets functions to act on a TensorNetworkCircuit interactively.\n\n\n\n\n\n","category":"type"},{"location":"backends/backend/#PicoQuant.save_tensor_data","page":"Backends","title":"PicoQuant.save_tensor_data","text":"save_tensor_data(backend::InteractiveBackend{T},\n                          tensor_label::Symbol,\n                          tensor_data::Array{<:Number}) where T <: AbstractArray\n\nFunction to save tensor data to a dictionary in an interactive backend.\n\n\n\n\n\nsave_tensor_data(backend::DSLBackend,\n                          node_label::Symbol,\n                          tensor_label::Symbol,\n                          tensor_data::Array{<:Number})\n\nFunction to save tensor data to the hdf5 file specified by a dsl backend. It also appends a load command to the dsl script contained in the backend to load the saved tensor data as a tensor with the given node label.\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#PicoQuant.load_tensor_data","page":"Backends","title":"PicoQuant.load_tensor_data","text":"load_tensor_data(backend::InteractiveBackend{T},\n                          tensor_label::Symbol) where T <: AbstractArray\n\nFunction to load tensor data from backend storage (if present)\n\n\n\n\n\nload_tensor_data(backend::DSLBackend,\n                          tensor_label::Symbol)\n\nFunction to load tensor data from backend storage (if present)\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#PicoQuant.save_output","page":"Backends","title":"PicoQuant.save_output","text":"save_output(backend::InteractiveBackend{T},\n                     node::Symbol, name::String) where T <: AbstractArray\n\nFunction to save the result of fully contracting a network under the given name.\n\n\n\n\n\nsave_output(backend::DSLBackend, node::Symbol, name::String)\n\nFunction to save the result of fully contracting a network under the given name.\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#PicoQuant.reshape_tensor","page":"Backends","title":"PicoQuant.reshape_tensor","text":"reshape_tensor(backend::InteractiveBackend{T},\n                        tensor::Symbol,\n                        groups::Array{Array{Int, 1}, 1}) where T <: AbstractArray\n\nFunction to reshape a given tensor.\n\n\n\n\n\nreshape_tensor(backend::DSLBackend,\n                        tensor::Symbol,\n                        groups::Array{Array{Int, 1}, 1})\n\nFunction to add dsl command that reshapes a given tensor.\n\n\n\n\n\nreshape_tensor(tensor::AbstractArray,\n                        dims::Union{Int, Array{Int,1}})\n\nReshape a tensor\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#PicoQuant.decompose_tensor!","page":"Backends","title":"PicoQuant.decompose_tensor!","text":"decompose_tensor!(backend::InteractiveBackend{T},\n                           tensor::Symbol,\n                           left_positions::Array{Int, 1},\n                           right_positions::Array{Int, 1};\n                           threshold::AbstractFloat=1e-13,\n                           max_rank::Int=0,\n                           left_label::Symbol,\n                           right_label::Symbol) where T <: AbstractArray\n\nFunction to decompose a single tensor into two tensors and return the dimension of the newly created virtual edge.\n\n\n\n\n\ndecompose_tensor!(backend::DSLBackend,\n                           tensor::Symbol,\n                           left_indices::Array{Int, 1},\n                           right_indices::Array{Int, 1};\n                           threshold::AbstractFloat=1e-13,\n                           max_rank::Int=0,\n                           left_label::Symbol,\n                           right_label::Symbol)\n\nFunction to decompose a single tensor into two tensors. Returns 0 as an indication that the dimension of the new virtual edge cannot be determined until runtime.\n\n\n\n\n\ndecompose_tensor!(tng::TensorNetworkCircuit,\n                           node::Symbol\n                           left_indices::Array{Symbol, 1},\n                           right_indices::Array{Symbol, 1};\n                           threshold::AbstractFloat=1e-15,\n                           max_rank::Int=0,\n                           left_label::Union{Nothing, Symbol}=nothing,\n                           right_label::Union{Nothing, Symbol}=nothing)\n\nDecompose a tensor into two smaller tensors\n\n\n\n\n\n","category":"function"},{"location":"backends/backend/#Base.push!","page":"Backends","title":"Base.push!","text":"push!(dsl::DSLBackend, instruction::String)\n\nFunction to append a dsl command to the dsl script contained in a dsl backend.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#Layer-1-Operations","page":"Layer 1","title":"Layer 1 Operations","text":"","category":"section"},{"location":"layers/layer1/","page":"Layer 1","title":"Layer 1","text":"Layer 1 operations perform the actual computations that are coordinated by layer 2. This layer consists of a number of backends which are capable of performing the computations for a variety of different platforms and scenarios.","category":"page"},{"location":"layers/layer1/#Title","page":"Layer 1","title":"Title","text":"","category":"section"},{"location":"layers/layer1/","page":"Layer 1","title":"Layer 1","text":"PicoQuant.load_tensor!\nPicoQuant.save_tensor!\nPicoQuant.delete_tensor!\nPicoQuant.contract_tensors\nPicoQuant.reshape_tensor\nPicoQuant.transpose_tensor\nPicoQuant.conjugate_tensor\nPicoQuant.execute_dsl_file","category":"page"},{"location":"layers/layer1/#PicoQuant.load_tensor!","page":"Layer 1","title":"PicoQuant.load_tensor!","text":"load_tensor!(tensors::Dict{Symbol, Array{T}},\n                      tensor_label::String,\n                      data_label::String,\n                      tensor_data_filename::String) where T <: Number\n\nLoad tensor data, identified by tensor_label, from a .h5 file and store it in the dictionary 'tensors'\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.save_tensor!","page":"Layer 1","title":"PicoQuant.save_tensor!","text":"save_tensor!(tensor_data_filename::String,\n             tensors::Dict{Symbol, <: AbstractArray},\n             tensor_label::String,\n             group_name::String=\"\")\n\nSave tensor data to a .h5 file. If no group name is given for the data the label that identifies the tensor in the dictionary 'tensors' is used.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.delete_tensor!","page":"Layer 1","title":"PicoQuant.delete_tensor!","text":"delete_tensor!(backend::InteractiveBackend{T}, tensor_label::Symbol) where T <: AbstractArray\n\nMark tensor for deletion\n\n\n\n\n\ndelete_tensor!(backend::DSLBackend, tensor_label::Symbol)\n\nMark tensor for deletion\n\n\n\n\n\ndelete_tensor!(tensors::Dict{Symbol, <: AbstractArray,\n                        tensor_label::String)\n\nDelete the specified tensor from the dictionary 'tensors'\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.contract_tensors","page":"Layer 1","title":"PicoQuant.contract_tensors","text":"contract_tensors(backend::InteractiveBackend{T},\n                          A_label::Symbol, A_ncon_indices::Array{Int, 1},\n                          B_label::Symbol, B_ncon_indices::Array{Int, 1},\n                          C_label::Symbol) where T <: AbstractArray\n\nFunction to interactively contract two tensors A and B to create a tensor C.\n\n\n\n\n\ncontract_tensors(backend::DSLBackend,\n                          A_label::Symbol, A_ncon_indices::Array{Int, 1},\n                          B_label::Symbol, B_ncon_indices::Array{Int, 1},\n                          C_label::Symbol)\n\nFunction to add dsl commands to a dsl backend's script that contract two tensors A and B to create a third tensor C.\n\n\n\n\n\ncontract_tensors(tensors_to_contract::Tuple{<: AbstractArray, <: AbstractArray},\n                 tensor_indices::Tuple{Array{Int,1},Array{Int,1}})\n\nFunction to contract the tensors contained in the tuple 'tensorstocontract' according to the ncon indices given and return the result.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.transpose_tensor","page":"Layer 1","title":"PicoQuant.transpose_tensor","text":"transpose_tensor(tensor::AbstractArray,\n                          index_permutation::Array{Int, 1})\n\nTranspose a tensor by permuting the indices as specified in index_permutation.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.conjugate_tensor","page":"Layer 1","title":"PicoQuant.conjugate_tensor","text":"function conjugate_tensor(tensor::AbstractArray)\n\nConjugate the elements of a tensor.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer1/#PicoQuant.execute_dsl_file","page":"Layer 1","title":"PicoQuant.execute_dsl_file","text":"execute_dsl_file(::Type{T}, dsl_filename::String=\"contract_network.tl\",\n                        tensor_data_filename::String=\"tensor_data.h5\",\n                        output_data_filename::String=\"\") where T <: AbstractArray\n\nFunction to read dsl commands from a given dsl file and execute them using tensor data from the given tensor data file. The type of tensor storage to use for the simulation is specified by the first argument\n\n\n\n\n\nexecute_dsl_file(dsl_filename::String=\"contract_network.tl\",\n                          tensor_data_filename::String=\"tensor_data.h5\",\n                          output_data_filename::String=\"\")\n\nFunction to read dsl commands from a given dsl file and execute them using tensor data from the given tensor data file. If the type of tensor storage to use is not specified, it defaults to Array{ComplexF32}\n\n\n\n\n\n","category":"function"},{"location":"visuals/#PicoQuant-Visualisation","page":"Visualisation","title":"PicoQuant Visualisation","text":"","category":"section"},{"location":"visuals/","page":"Visualisation","title":"Visualisation","text":"It can be useful to create visualisations of the tensor network circuit to get insights into the structure and find issues with their manipulation. Some useful functions for this are listed here.","category":"page"},{"location":"visuals/","page":"Visualisation","title":"Visualisation","text":"PicoQuant.create_graph\nPicoQuant.plot","category":"page"},{"location":"visuals/#PicoQuant.create_graph","page":"Visualisation","title":"PicoQuant.create_graph","text":"function create_graph(tng::TensorNetworkCircuit)\n\nCreate a light graph representation of the tensor network circuit\n\n\n\n\n\n","category":"function"},{"location":"visuals/#PicoQuant.plot","page":"Visualisation","title":"PicoQuant.plot","text":"function plot(tng::TensorNetworkCircuit; showlabels::Bool=false)\n\nFunction for creating a graph of a tensor network graph\n\n\n\n\n\n","category":"function"},{"location":"algo/algos/#Sample-algorithms","page":"Algorithms","title":"Sample algorithms","text":"","category":"section"},{"location":"algo/algos/#Title","page":"Algorithms","title":"Title","text":"","category":"section"},{"location":"algo/algos/","page":"Algorithms","title":"Algorithms","text":"PicoQuant.create_ghz_preparation_circuit\nPicoQuant.create_qft_circuit\nPicoQuant.create_simple_preparation_circuit","category":"page"},{"location":"algo/algos/#PicoQuant.create_ghz_preparation_circuit","page":"Algorithms","title":"PicoQuant.create_ghz_preparation_circuit","text":"create_ghz_preparation_circuit(qubits::Int)\n\nCreate a ghz preparation circuit\n\n\n\n\n\n","category":"function"},{"location":"algo/algos/#PicoQuant.create_qft_circuit","page":"Algorithms","title":"PicoQuant.create_qft_circuit","text":"create_qft_circuit(n::Int, m::Int=-1)\n\nGenerate QFT circuit acting on n qubits with approximation parameter m. To no approximation m is set to -1 which is the default value.\n\n\n\n\n\n","category":"function"},{"location":"algo/algos/#PicoQuant.create_simple_preparation_circuit","page":"Algorithms","title":"PicoQuant.create_simple_preparation_circuit","text":"create_simple_preparation_circuit(qubits::Int,\n                                           depth::Int,\n                                           seed::Int=nothing)\n\nCreate a simple preparation circuit which mixes\n\n\n\n\n\n","category":"function"},{"location":"#PicoQuant.jl-Documentation","page":"Home","title":"PicoQuant.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PicoQuant.jl is a","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"tutorial/#PicoQuant.jl-Tutorial","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"Author = \"QuantEx Team\"","category":"page"},{"location":"tutorial/#Tutorial","page":"PicoQuant.jl Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"PicoQuant.jl is a ...","category":"page"},{"location":"tutorial/","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"using PicoQuant\n","category":"page"},{"location":"tutorial/#Topics","page":"PicoQuant.jl Tutorial","title":"Topics","text":"","category":"section"},{"location":"tutorial/","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"When...:","category":"page"},{"location":"tutorial/","page":"PicoQuant.jl Tutorial","title":"PicoQuant.jl Tutorial","text":"x = 1","category":"page"},{"location":"layers/layer3/#Layer-3-Operations","page":"Layer 3","title":"Layer 3 Operations","text":"","category":"section"},{"location":"layers/layer3/","page":"Layer 3","title":"Layer 3","text":"Layer 3 is the highest layer and deals with the conversion of circuit descriptions (QASM/qiskit circuit objects) to tensor networks. It provides data structures and functionality to represent and manipulate tensor networks representing quantum circuits.","category":"page"},{"location":"layers/layer3/#Tensor-network-fundamentals","page":"Layer 3","title":"Tensor network fundamentals","text":"","category":"section"},{"location":"layers/layer3/","page":"Layer 3","title":"Layer 3","text":"PicoQuant.TensorNetworkCircuit\nPicoQuant.Node\nPicoQuant.Edge\nPicoQuant.add_gate!\nPicoQuant.edges\nPicoQuant.new_label!\nPicoQuant.add_input!\nPicoQuant.add_output!\nPicoQuant.inneighbours\nPicoQuant.outneighbours\nPicoQuant.virtualneighbours\nPicoQuant.neighbours\nPicoQuant.inedges\nPicoQuant.outedges\nPicoQuant.decompose_gate!","category":"page"},{"location":"layers/layer3/#PicoQuant.TensorNetworkCircuit","page":"Layer 3","title":"PicoQuant.TensorNetworkCircuit","text":"Struct for tensor network graph of a circuit\n\n\n\n\n\n","category":"type"},{"location":"layers/layer3/#PicoQuant.Node","page":"Layer 3","title":"PicoQuant.Node","text":"Struct to represent a node in tensor network graph\n\n\n\n\n\n","category":"type"},{"location":"layers/layer3/#PicoQuant.Edge","page":"Layer 3","title":"PicoQuant.Edge","text":"Struct to represent an edge\n\n\n\n\n\n","category":"type"},{"location":"layers/layer3/#PicoQuant.add_gate!","page":"Layer 3","title":"PicoQuant.add_gate!","text":"add_gate!(network::TensorNetworkCircuit,\n                   gate_data::Array{<:Number},\n                   targetqubits::Array{Int, 1};\n                   decompose::Bool=false)\n\nAdd a node to the tensor network for the given gate acting on the given quibits\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.edges","page":"Layer 3","title":"PicoQuant.edges","text":"edges(network::TensorNetworkCircuit)\n\nFunction to return edges from a given network\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.new_label!","page":"Layer 3","title":"PicoQuant.new_label!","text":"new_label!(network::TensorNetworkCircuit, label_str)\n\nFunction to create a unique symbol by incrememting relevant counter\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.add_input!","page":"Layer 3","title":"PicoQuant.add_input!","text":"add_input!(network::TensorNetworkCircuit, config::String)\n\nFunction to add input nodes to a tensor network circuit with a given input configuration\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.add_output!","page":"Layer 3","title":"PicoQuant.add_output!","text":"add_output!(network::TensorNetworkCircuit, config::String)\n\nFunction to add output nodes to a tensor network circuit with a given output configuration\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.inneighbours","page":"Layer 3","title":"PicoQuant.inneighbours","text":"inneighbours(network::TensorNetworkCircuit,\n                      node_label::Symbol)\n\nFunction to return the nodes which are connected to the given node with incoming edges\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.outneighbours","page":"Layer 3","title":"PicoQuant.outneighbours","text":"outneighbours(network::TensorNetworkCircuit,\n                       node_label::Symbol)\n\nFunction to return the nodes which are connected to the given node with outgoing edges\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.virtualneighbours","page":"Layer 3","title":"PicoQuant.virtualneighbours","text":"virtualneighbours(network::TensorNetworkCircuit,\n                       node_label::Symbol)\n\nFunction to return the nodes which are connected to the given node with virtual edges\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.neighbours","page":"Layer 3","title":"PicoQuant.neighbours","text":"neighbours(network::TensorNetworkCircuit,\n\nFunction to get all neighbouring nodes of the given node (incoming, outgoing and virtual)\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.inedges","page":"Layer 3","title":"PicoQuant.inedges","text":"inedges(network::TensorNetworkCircuit,\n                 node_label::Symbol)\n\nFunction to get all incoming edges to the current node\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.outedges","page":"Layer 3","title":"PicoQuant.outedges","text":"outedges(network::TensorNetworkCircuit,\n                  node_label::Symbol)\n\nFunction to get all outgoing edges to the current node\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.decompose_gate!","page":"Layer 3","title":"PicoQuant.decompose_gate!","text":"decompose_gate!(gate_data::Array{<:Number, 4},\n                         threshold::AbstractFloat=1e-15)\n\nDecompose a tensor into two smaller tensors\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#Qiskit/external-package-interoperability","page":"Layer 3","title":"Qiskit/external package interoperability","text":"","category":"section"},{"location":"layers/layer3/","page":"Layer 3","title":"Layer 3","text":"PicoQuant.load_qasm_as_circuit_from_file\nPicoQuant.load_qasm_as_circuit\nPicoQuant.convert_qiskit_circ_to_network","category":"page"},{"location":"layers/layer3/#PicoQuant.load_qasm_as_circuit_from_file","page":"Layer 3","title":"PicoQuant.load_qasm_as_circuit_from_file","text":"load_qasm_as_circuit_from_file(qasm_path::String)\n\nFunction to load qasm from the given path and return a qiskit circuit\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.load_qasm_as_circuit","page":"Layer 3","title":"PicoQuant.load_qasm_as_circuit","text":"load_qasm_as_circuit(qasm_str::String)\n\nFunction to load qasm from a given qasm string and return a qiskit circuit\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.convert_qiskit_circ_to_network","page":"Layer 3","title":"PicoQuant.convert_qiskit_circ_to_network","text":"convert_qiskit_circ_to_network(circ, backend::AbstractBackend=InteractiveBackend();\n                                        decompose::Bool=false,\n                                        transpile::Bool=false)\n\nGiven a qiskit circuit object, this function will convert this to a tensor network circuit with a backend set to backend. If the decompose option is true it will decompose two qubit gates to two tensors acting on each qubit with a virtual bond connecting them. The transpile option will transpile the circuit using the basicswap pass from qiskit to ensure that two qubit gates are only applied between neighbouring qubits.\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#Circuit-I/O-transformers","page":"Layer 3","title":"Circuit I/O transformers","text":"","category":"section"},{"location":"layers/layer3/","page":"Layer 3","title":"Layer 3","text":"PicoQuant.to_dict\nPicoQuant.to_json\nPicoQuant.network_from_dict\nPicoQuant.edge_from_dict\nPicoQuant.node_from_dict\nPicoQuant.network_from_json","category":"page"},{"location":"layers/layer3/#PicoQuant.to_dict","page":"Layer 3","title":"PicoQuant.to_dict","text":"to_dict(edge::Edge)\n\nFunction to convert an edge instance to a serialisable dictionary\n\n\n\n\n\nto_dict(node::Node)\n\nFunction to serialise node instance to json format\n\n\n\n\n\nto_dict(network::TensorNetworkCircuit)\n\nConvert a tensor network to a nested dictionary\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.to_json","page":"Layer 3","title":"PicoQuant.to_json","text":"to_json(tng::TensorNetworkCircuit)\n\nConvert a tensor network to a json string\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.network_from_dict","page":"Layer 3","title":"PicoQuant.network_from_dict","text":"network_from_dict(dict::Dict{String, Any}, backend::AbstractBackend=InteractiveBackend())\n\nConvert a dictionary to a tensor network with a given backend (default: InteractiveBackend)\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.edge_from_dict","page":"Layer 3","title":"PicoQuant.edge_from_dict","text":"edge_from_dict(dict::Dict)\n\nFunction to create an edge instance from a dictionary\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.node_from_dict","page":"Layer 3","title":"PicoQuant.node_from_dict","text":"node_from_dict(d::Dict)\n\nFunction to create a node instance from a json string\n\n\n\n\n\n","category":"function"},{"location":"layers/layer3/#PicoQuant.network_from_json","page":"Layer 3","title":"PicoQuant.network_from_json","text":"network_from_json(json_str::String, backend::AbstractBackend=InteractiveBackend())\n\nConvert a json string to a tensor network\n\n\n\n\n\n","category":"function"}]
}
