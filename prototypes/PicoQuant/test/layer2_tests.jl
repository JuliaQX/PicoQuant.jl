using HDF5

@testset "Test building of contraction plans and loading/savingn json" begin

    qasm_str = """OPENQASM 2.0;
                  include "qelib1.inc";
                  qreg q[3];
                  h q[0];
                  cx q[0],q[1];
                  cx q[1],q[2];"""

    circ = load_qasm_as_circuit(qasm_str)
    InteractiveBackend()
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
    DSLWriter()
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "000")
    add_output!(tng, "000")
    plan = random_contraction_plan(tng)

    try
        # Test if contract_network creates files with dsl and tensor data.
        contract_network!(tng, plan)
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
        contract_network!(tng, plan, "vector")

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
    InteractiveBackend()
    tng = convert_qiskit_circ_to_network(circ)
    add_input!(tng, "00")
    plan = random_contraction_plan(tng)

    executer = InteractiveExecuter()
    contract_network!(tng, plan, executer)

    @test length(tng.nodes) == 1

    result_label = collect(values(tng.nodes))[1].data_label
    result = PicoQuant.backend.tensors[result_label]
    @test real(result)[1] ≈ 1/2
end

@testset "Test tensor chain compression" begin
    @test begin
        tn = TensorNetworkCircuit(2)
        add_input!(tn, "00")
        env = InteractiveExecuter()
        compress_tensor_chain!(tn, collect(keys(tn.nodes)), env)
        length(tn.nodes) == 2
    end

    @test begin
        # create a network and apply a ghz state preparation
        # circuit. Then apply two additional cnot gates which should
        # cancel out. At this point there should be 3 virtual bonds
        # each with dimension 2. After a compression operation, this
        # will be reduced to a single bond with dimension 2
        qasm_str = """OPENQASM 2.0;
                      include "qelib1.inc";
                      qreg q[2];
                      h q[0];
                      cx q[0],q[1];
                      cx q[0],q[1];
                      cx q[0],q[1];
                      """
        circ = load_qasm_as_circuit(qasm_str)
        tn = convert_qiskit_circ_to_network(circ)
        add_input!(tn, "0"^2)
        decompose_tensor!(tn, :node_2, [:index_3, :index_4], [:index_2, :index_5])
        decompose_tensor!(tn, :node_3, [:index_4, :index_6], [:index_5, :index_7])
        decompose_tensor!(tn, :node_4, [:index_6, :index_8], [:index_7, :index_9])

        # contract all links except the virtual and open edges
        plan = [k for (k, v) in pairs(tn.edges) if !v.virtual &&
                                                v.src != nothing &&
                                                v.dst != nothing]

        env = InteractiveExecuter()
        for edge in plan
              contract_pair!(tn, edge, env)
        end

        compress_tensor_chain!(tn, [:node_18, :node_19], env)

        idx = findfirst(x -> x == :index_14, tn.nodes[:node_18].indices)
        size(tn.nodes[:node_18].data)[idx] == 2
    end
end
