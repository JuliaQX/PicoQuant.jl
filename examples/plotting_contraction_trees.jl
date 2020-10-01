
using PicoQuant

# Create a tensor network circuit to find contraction plans for.
qasm_str = """OPENQASM 2.0;
              include "qelib1.inc";
              qreg q[3];
              h q[0];
              cx q[0],q[1];
              cx q[1],q[2];
              h q[2];
              cx q[0],q[1];
              cx q[0],q[2];"""

circ = load_qasm_as_circuit(qasm_str)
tng = convert_qiskit_circ_to_network(circ)
add_input!(tng, "000")
add_output!(tng, "000")

# %% Plot the optimal contraction tree
optimal_plan = netcon(tng)
optimal_contraction_tree = create_contraction_tree(tng, optimal_plan)

display(plot_contraction_tree(optimal_contraction_tree))

# %% Plot the random contration tree found by bgreedy
bgreedy_plan, time_cost = bgreedy(tng, 0.5, 2.5, 10)
bgreedy_contraction_tree = create_contraction_tree(tng, bgreedy_plan)

display(plot_contraction_tree(bgreedy_contraction_tree))
