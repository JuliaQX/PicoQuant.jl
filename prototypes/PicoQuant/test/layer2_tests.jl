@testset "Test building of contraction plaths and loading/savingn json" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    path = random_contraction_path(tng)

    # Is the length of the path equal the number of edges in the network?
    @test length(path) == 8

    # Are there no edges connecting a node to itself?
    @test begin
        no_self_connections = true
        for edge in path
            no_self_connections = no_self_connections && length(Set(edge)) == 2
        end
        no_self_connections
    end

    # Are there no repeated edges?
    @test begin
        no_repeats = true
        for edge in path
            path_edges = [Set(edge) for edge in path]
            no_repeats = no_repeats && length(path) == length(Set(path_edges))
        end
        no_repeats
    end

    # Test loading and saving path as json string
    @test begin
        path_json = contraction_path_to_json(path)
        path == contraction_path_from_json(path_json)
    end
end
