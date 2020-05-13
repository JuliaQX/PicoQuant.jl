@testset "Test loading and conversion of qasm" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    tng_json = to_json(tng)

    @test begin
        circ.n_qubits == 3
    end

    @test begin
        length(circ.data) == 3
    end

    @test begin
        # test conversion to json and back maintains graph
        tng_to_json_from_json = to_json(network_from_json(tng_json))
        tng_to_json_from_json == tng_json
    end
end

@testset "Test tensor network data structure" begin
    tn = TensorNetworkCircuit(3)

    @test begin
        # Check if tn has the correct number of nodes
        length(tn.nodes) == 0
    end

    hadamard = [1 1; 1 -1]./sqrt(2)
    add_gate!(tn, hadamard, [1])
    @test begin
        length(tn.nodes) == 1
    end

    @test begin
        length(edges(tn)) == 4
    end
end
