@testset "Test contracting a network" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    path = random_contraction_path(tng)
    contract_network!(tng, path)

    # Is there only one node left after the contraction?
    @test begin
        all_equal = true
        for i = 2:length(tng.nodes)
            all_eual = all_equal && tng.nodes[1] == tng.nodes[i]
        end
        all_equal
    end

    # Create a network which will be disjoint when only edges are contracted
    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[2];
                  h q[0];
                  h q[1];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    path = random_contraction_path(tng)
    contract_network!(tng, path)

    # Is there only one node left after the contraction?
    @test begin
        all_equal = true
        for i = 2:length(tng.nodes)
            all_eual = all_equal && tng.nodes[1] == tng.nodes[i]
        end
        all_equal
    end
end
