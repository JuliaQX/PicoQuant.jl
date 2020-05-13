# # Hello Pico Quant Example
# This is an example to demonstrate basic interface and features of the
# PicoQuant package.

using PicoQuant

# We will test with a very simple circuit, the GHZ preparation circuit for
# 3 qubits. We define this circuit by craeting a string containing its qasm
# representation.

qasm_str = """OPENQASM 2.0;
              include "qelib1.inc";
              qreg q[3];
              h q[0];
              cx q[0],q[1];
              cx q[1],q[2];"""

# Next we load this and convert to a tensor network representation

## Use qiskit to load the qasm and get a qiskit circuit object
circ = load_qasm_as_circuit(qasm_str)

## From qiskit circuit object build a tensor network representation
tng = convert_qiskit_circ_to_network(circ)

# This prepares a tensor network of the circuit gates but does not define
# input qubits or values for these. We wish to add all 0's as

add_input!(tng, "000")

# Create a random contraction plan by randomly selecting an ordering of edges

plan = random_contraction_plan(tng)

# Create a dsl_writer object which is used to write DSL commands and save
# tensor data to the given files.

executer = dsl_writer("ghz_3.tl", "ghz_3_data.h5")

# Using the given contraction plan, we prepare a DSL description which will
# execute the given plan and also write the tensors required to a file on disk.

contract_network!(tng, plan, executer, "vector")

# We can examine the DSL that is produced

;cat "ghz_3.tl"

# and also execute the DSL to produce the desired output.

execute_dsl_file("ghz_3.tl", "ghz_3_data.h5")
