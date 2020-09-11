import MetaGraphs

@testset "Visualisation tests" begin
    @testset "Test contracting a network" begin

        qasm_str = """OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q[3];
                    h q[0];
                    cx q[0],q[1];
                    cx q[1],q[2];"""

        circ = load_qasm_as_circuit(qasm_str)
        tng = convert_qiskit_circ_to_network(circ, InteractiveBackend())
        add_input!(tng, "000")
        add_output!(tng, "000")
        g = PicoQuant.create_graph(tng)

        # Is there only one node left after the contraction?
        @test begin
            MetaGraphs.nv(g) == 9
        end

        @test begin
            MetaGraphs.ne(g) == 8
        end
    end
end