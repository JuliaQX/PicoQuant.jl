@testset "Test loading and conversion of qasm" begin
    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""
    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_to_tensor_network_graph(circ)
    tng_json = to_json(tng)
    @test begin
        circ.n_qubits == 3
    end

    @test begin
        length(circ.data) == 3
    end

    @test begin
        # test conversion to json and back maintains graph
        tng_to_json_from_json = to_json(tng_from_json(tng_json))
        tng_to_json_from_json == tng_json
    end
end
