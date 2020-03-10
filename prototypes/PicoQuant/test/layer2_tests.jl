using LightGraphs

@testset "Test building of contraction plan and loading and saving" begin
    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""
    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_to_tensor_network_graph(circ)
    plan = simple_contraction_plan(tng)
    @test begin
        length(plan.edges) == length(edges(tng.graph))
    end

    @test begin
        plan_json = to_json(plan)
        plan == contraction_plan_from_json(plan_json)
        # plan.edges == contraction_plan_from_json(plan_json).edges
    end
end
