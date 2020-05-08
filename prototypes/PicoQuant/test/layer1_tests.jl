@testset "Test contracting a network" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "000")
    add_output!(tng, "000")
    plan = random_contraction_plan(tng)
    contract_network!(tng, plan)

    # Is there only one node left after the contraction?
    @test begin
        length(tng.nodes) == 1
    end

    # Check if the correct amplitube was computed
    @test begin
        node, _ = iterate(tng.nodes)
        node.second.data[1] ≈ 1/sqrt(2)
    end

    # Create a network which will be disjoint when only edges are contracted
    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[2];
                  h q[0];
                  h q[1];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "00")
    add_output!(tng, "00")
    plan = random_contraction_plan(tng)
    contract_network!(tng, plan)

    # Is there only one node left in nodes array after the contraction?
    @test begin
        length(tng.nodes) == 1
    end
end
