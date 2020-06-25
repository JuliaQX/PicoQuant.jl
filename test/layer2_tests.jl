using HDF5


"""
    function dsl_backend_teardown()

Function to cleanup after the dslbackend
"""
function dsl_backend_teardown()
    rm("contract_network.tl", force=true)
    rm("tensor_data.h5", force=true)
end

"""Dictionary of init and finalise methods for testing backends"""
backends = Dict("DSLBackend" =>
                Dict(:init => DSLBackend,
                     :execute => execute_dsl_file,
                     :finalise => dsl_backend_teardown),

                "InteractiveBackend" =>
                Dict(:init => InteractiveBackend,
                     :execute => () -> (),
                     :finalise => () -> ())
                )


for (backend_name, backend_funcs) in pairs(backends)
    @testset "Test building, saving and loading contraction plans with $backend_name" begin

        qasm_str = """OPENQASM 2.0;
                      include "qelib1.inc";
                      qreg q[3];
                      h q[0];
                      cx q[0],q[1];
                      cx q[1],q[2];"""

        try
            backend_funcs[:init]()
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
        finally
            backend_funcs[:finalise]()
        end
    end


    @testset "Test contract functions for $backend_name" begin

        # Create a tensor network and contraction plan to work with.
        qasm_str = """OPENQASM 2.0;
                      include "qelib1.inc";
                      qreg q[3];
                      h q[0];
                      cx q[0],q[1];
                      cx q[1],q[2];"""

        try
            backend_funcs[:init]()
            circ = load_qasm_as_circuit(qasm_str)
            tng = convert_qiskit_circ_to_network(circ)
            add_input!(tng, "000")
            add_output!(tng, "000")
            plan = random_contraction_plan(tng)

            # Test if contract_network creates files with dsl and tensor data.
            contract_network!(tng, plan)

            backend_funcs[:execute]

            # Is there only one node left after the contraction?
            @test begin
                length(tng.nodes) == 1
            end
        finally
            backend_funcs[:finalise]()
        end

        try
            backend_funcs[:init]()
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

            contract_network!(tng, plan, "vector")

            # Is there only one node left in nodes array after the contraction?
            @test begin
                length(tng.nodes) == 1
            end

            backend_funcs[:execute]()

            # Check if the final tensor is a vector
            @test begin
                result = load_tensor_data(backend, :result)
                length(size(result)) == 1
            end

            @test begin
                result = load_tensor_data(backend, :result)
                real(result)[1] ≈ 1/2
            end
        finally
            backend_funcs[:finalise]()
        end
    end

    @testset "Test full wavefunction contraction for $backend_name" begin

        # Create a tensor network to work with.
        qasm_str = """OPENQASM 2.0;
                      include "qelib1.inc";
                      qreg q[3];
                      h q[0];
                      cx q[0],q[1];
                      cx q[0],q[1];
                      cx q[0],q[2];
                      """

        try
            backend_funcs[:init]()
            circ = load_qasm_as_circuit(qasm_str)
            tn = convert_qiskit_circ_to_network(circ)
            add_input!(tn, "000")

            # See if full wavefunction contraction completes.
            full_wavefunction_contraction!(tn, "vector")

            backend_funcs[:execute]

            # Is there only one node left after the contraction?
            @test begin
                length(tn.nodes) == 1
            end
        finally
            backend_funcs[:finalise]()
        end
    end

    @testset "Test tensor chain compression for $backend_name" begin

        @test begin
            try
                backend_funcs[:init]()
                tn = TensorNetworkCircuit(2)
                add_input!(tn, "00")
                compress_tensor_chain!(tn, collect(keys(tn.nodes)))
                length(tn.nodes) == 2
            finally
                backend_funcs[:finalise]()
            end
        end

        @testset begin

            try
                backend_funcs[:init]()
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
                InteractiveBackend()
                circ = load_qasm_as_circuit(qasm_str)
                tn = convert_qiskit_circ_to_network(circ)
                add_input!(tn, "0"^2)
                decompose_tensor!(tn, :node_2, [:index_3, :index_4],
                                  [:index_2, :index_5])
                decompose_tensor!(tn, :node_3, [:index_4, :index_6],
                                  [:index_5, :index_7])
                decompose_tensor!(tn, :node_4, [:index_6, :index_8],
                                  [:index_7, :index_9])

                # contract all links except the virtual and open edges
                plan = [k for (k, v) in pairs(tn.edges) if !v.virtual &&
                                                        v.src != nothing &&
                                                        v.dst != nothing]

                for edge in plan
                      contract_pair!(tn, edge)
                end

                compress_tensor_chain!(tn, [:node_18, :node_19])

                idx = findfirst(x -> x == :index_14, tn.nodes[:node_18].indices)

                backend_funcs[:execute]()

                result = load_tensor_data(backend, :node_18)

                size(result)[idx] == 2
            finally
                backend_funcs[:finalise]()
            end
        end
    end

    @testset "Test tensor network contraction with MPS methods for $backend_name" begin
        @test begin
            try
                backend_funcs[:init]()
                n = 5
                circ = create_ghz_preparation_circuit(n)
                tn = convert_qiskit_circ_to_network(circ,
                                                    decompose=true,
                                                    transpile=true)
                add_input!(tn, "0"^n)
                mps_nodes = contract_mps_tensor_network_circuit!(tn)
                calculate_mps_amplitudes!(tn, mps_nodes)
                backend_funcs[:execute]()
                ref_output = zeros(ComplexF64, 2^n)
                ref_output[[1, end]] .= 1/sqrt(2)
                load_tensor_data(backend, :result) ≈ ref_output
            finally
                backend_funcs[:finalise]()
            end
        end
    end
end
