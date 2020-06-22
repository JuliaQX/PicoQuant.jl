# Layer 3 Operations

Layer 3 is the highest layer and deals with the conversion of circuit
descriptions (QASM/qiskit circuit objects) to tensor networks. It provides
data structures and functionality to represent and manipulate tensor networks
representing quantum circuits.

## Tensor network fundamentals

```@docs
PicoQuant.TensorNetworkCircuit
PicoQuant.Node
PicoQuant.Edge
PicoQuant.add_gate!
PicoQuant.edges
PicoQuant.new_label!
PicoQuant.add_input!
PicoQuant.add_output!
PicoQuant.inneighbours
PicoQuant.outneighbours
PicoQuant.virtualneighbours
PicoQuant.neighbours
PicoQuant.inedges
PicoQuant.outedges
PicoQuant.decompose_gate!
```

## Qiskit/external package interoperability
```@docs
PicoQuant.load_qasm_as_circuit_from_file
PicoQuant.load_qasm_as_circuit
PicoQuant.convert_qiskit_circ_to_network
```

## Circuit I/O transformers
```@docs
PicoQuant.to_dict
PicoQuant.to_json
PicoQuant.network_from_dict
PicoQuant.edge_from_dict
PicoQuant.node_from_dict
PicoQuant.network_from_json
```
