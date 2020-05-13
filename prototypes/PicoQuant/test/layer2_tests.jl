using HDF5

@testset "Test building of contraction plans and loading/savingn json" begin

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

    # Test loading and saving plan as json string.
    @test begin
        plan_json = contraction_plan_to_json(plan)
        plan == contraction_plan_from_json(plan_json)
    end
end

@testset "Test contract functions" begin

    # Create a tensor network and contraction plan to work with.
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

    executer = DSLWriter()

    try
        # Test if contract_network creates files with dsl and tensor data.
        contract_network!(tng, plan, executer)
        @test isfile("contract_network.tl")
        @test isfile("tensor_data.h5")

        # Is there only one node left after the contraction?
        @test begin
            length(tng.nodes) == 1
        end
    finally
        # Clean up any files created.
        if isfile("contract_network.tl")
            rm("contract_network.tl")
        end

        if isfile("tensor_data.h5")
            rm("tensor_data.h5")
        end
    end

    # Create a network which will be disjoint when all edges are contracted
    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[2];
                  h q[0];
                  h q[1];"""

    circ = load_qasm_as_circuit(qasm_str)
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "00")
    plan = random_contraction_plan(tng)

    try
        contract_network!(tng, plan, executer, "vector")

        # Is there only one node left in nodes array after the contraction?
        @test begin
            length(tng.nodes) == 1
        end

        execute_dsl_file("contract_network.tl", "tensor_data.h5")

        # Check if the final tensor is a vector
        @test begin
            result = h5open("tensor_data.h5", "r") do file
                read(file, "result")
            end
            length(size(result)) == 1
        end
    finally
        # Clean up any files created.
        if isfile("contract_network.tl")
            rm("contract_network.tl")
        end

        if isfile("tensor_data.h5")
            rm("tensor_data.h5")
        end
    end

    # Create the network again to test the interactive executer
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "00")
    plan = random_contraction_plan(tng)

    executer = InteractiveExecutor()
    contract_network!(tng, plan, executer)

    @test length(tng.nodes) == 1

    result = collect(values(tng.nodes))[1].data
    @test real(result)[1] ≈ 1/2
end
