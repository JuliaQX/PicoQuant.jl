@testset "Test building of contraction plaths and loading/savingn json" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    plan = random_contraction_plan(tng)

    # Is the length of the plan equal the number of closed edges in the network?
    @test length(plan) == 2

    # Are there no edges connecting a node to itself?
    @test begin
        no_self_connections = true
        for edge_label in plan
            edge = tng.edges[edge_label]
            # if edge ends not null
            if !(edge.src == edge.dst == nothing)
                if edge.src == edge.dst
                    no_self_connections = false
                end
            end
        end
        no_self_connections
    end

    # Are there no repeated edges in the plan?
    @test begin
        length(Set(plan)) == length(plan)
    end

    # Test loading and saving plan as json string
    @test begin
        plan_json = contraction_plan_to_json(plan)
        plan == contraction_plan_from_json(plan_json)
    end
end
