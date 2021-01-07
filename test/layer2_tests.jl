module Layer2Tests

using Test
using HDF5
using RandomMatrices
using LinearAlgebra

using PicoQuant

include("test_utils.jl")

@testset "Layer 2 tests" begin
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
                    Dict(:init => () -> DSLBackend("contract_network.tl", "tensor_data.h5", "", true),
                        # :execute => (args...) -> execute_dsl_file(ComplexF64, args...),
                        :execute => execute_dsl_file,
                        :finalise => dsl_backend_teardown),

                    "InteractiveBackend" =>
                    Dict(:init => InteractiveBackend,
                        :execute => () -> (),
                        :finalise => () -> ())
                    )


    for (backend_name, backend_funcs) in pairs(backends)

        @testset "Test contract functions for $backend_name" begin

            # Create a tensor network and contraction plan to work with.
            qasm_str = """OPENQASM 2.0;
                        include "qelib1.inc";
                        qreg q[3];
                        h q[0];
                        cx q[0],q[1];
                        cx q[1],q[2];"""

            try
                circ = load_qasm_as_circuit(qasm_str)
                tng = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
                add_input!(tng, "000")
                add_output!(tng, "000")
                plan = random_contraction_plan(tng)

                # Test if contract_network creates files with dsl and tensor data.
                contract_network!(tng, plan)

                backend_funcs[:execute]()

                # Is there only one node left after the contraction?
                @test begin
                    length(tng.nodes) == 1
                end
            finally
                backend_funcs[:finalise]()
            end

            try
                # Create a network which will be disjoint when all edges are contracted
                qasm_str = """OPENQASM 2.0;
                            include "qelib1.inc";
                            qreg q[2];
                            h q[0];
                            h q[1];"""

                circ = load_qasm_as_circuit(qasm_str)
                tng = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
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
                    result = load_tensor_data(tng, :result)
                    length(size(result)) == 1
                end

                @test begin
                    result = load_tensor_data(tng, :result)
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
                circ = load_qasm_as_circuit(qasm_str)
                tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
                add_input!(tn, "000")

                # See if full wavefunction contraction completes.
                full_wavefunction_contraction!(tn, "vector")

                # check if the computed metrics are correct
                @test begin
                    max_size = tn.backend.metrics.max_tensor_size == 8
                    total_space = tn.backend.metrics.total_space_allocated == 44
                    flops = tn.backend.metrics.flops == 124
                    max_size && total_space && flops
                end

                backend_funcs[:execute]()

                # Is there only one node left after the contraction?
                @test begin
                    length(tn.nodes) == 1
                end
            finally
                backend_funcs[:finalise]()
            end
        end

        @testset "Test netcon contraction function for $backend_name" begin
            # Create a tensor network to work with.
            qasm_str = """OPENQASM 2.0;
                        include "qelib1.inc";
                        qreg q[3];
                        h q[0];
                        cx q[0],q[1];
                        cx q[0],q[2];
                        """

            try
                circ = load_qasm_as_circuit(qasm_str)
                tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
                add_input!(tn, "000")
                add_output!(tn, "000")

                # See if full wavefunction contraction completes.
                netcon_contraction!(tn)

                backend_funcs[:execute]()

                # Is there only one node left after the contraction?
                @test begin
                    length(tn.nodes) == 1
                end
            finally
                backend_funcs[:finalise]()
            end
        end

        @testset "Test bgreedy contraction function for $backend_name" begin
            # Create a tensor network to work with.
            qasm_str = """OPENQASM 2.0;
                        include "qelib1.inc";
                        qreg q[3];
                        h q[0];
                        cx q[0],q[1];
                        cx q[0],q[2];
                        """

            try
                circ = load_qasm_as_circuit(qasm_str)
                tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
                add_input!(tn, "000")
                add_output!(tn, "000")

                # See if full wavefunction contraction completes.
                bgreedy_contraction!(tn)

                backend_funcs[:execute]()

                # Is there only one node left after the contraction?
                @test begin
                    length(tn.nodes) == 1
                end
            finally
                backend_funcs[:finalise]()
            end
        end

        @testset "Test full wavefunction and MPS contractions match for $backend_name" begin

            try
                circ = create_simple_preparation_circuit(3, 1).compose(create_ghz_preparation_circuit(3))

                # simulate with full wf approach
                tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init](), decompose=true, transpile=true)
                add_input!(tn, "0"^tn.number_qubits)

                # See if full wavefunction contraction completes.
                full_wavefunction_contraction!(tn, "vector")

                backend_funcs[:execute]()

                full_wf = load_tensor_data(tn, :result)

                tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init](), decompose=true, transpile=true)
                add_input!(tn, "0"^tn.number_qubits)

                # See if full wavefunction contraction completes.
                mps_nodes = contract_mps_tensor_network_circuit!(tn)
                calculate_mps_amplitudes!(tn, mps_nodes)

                backend_funcs[:execute]()

                mps_wf = load_tensor_data(tn, :result)

                # Is there only one node left after the contraction?
                @test begin
                    mps_wf ≈ full_wf
                end
            finally
                backend_funcs[:finalise]()
            end
        end

        @testset "Test tensor chain compression for $backend_name" begin

            @test begin
                try
                    tn = TensorNetworkCircuit(2, backend_funcs[:init]())
                    add_input!(tn, "00")
                    compress_tensor_chain!(tn, collect(keys(tn.nodes)))
                    length(tn.nodes) == 2
                finally
                    backend_funcs[:finalise]()
                end
            end

            @testset begin

                try
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
                    tn = convert_qiskit_circ_to_network(circ, backend_funcs[:init]())
                    add_input!(tn, "0"^2)
                    decompose_tensor!(tn, :node_2, [:index_3, :index_4],
                                    [:index_2, :index_5])
                    decompose_tensor!(tn, :node_3, [:index_4, :index_6],
                                    [:index_5, :index_7])
                    decompose_tensor!(tn, :node_4, [:index_6, :index_8],
                                    [:index_7, :index_9])

                    # contract all links except the virtual and open edges
                    plan = [k for (k, v) in pairs(tn.edges) if !v.virtual &&
                                                            v.src !== nothing &&
                                                            v.dst !== nothing]

                    for edge in plan
                        contract_pair!(tn, edge)
                    end

                    compress_tensor_chain!(tn, [:node_18, :node_19])

                    idx = findfirst(x -> x == :index_14, tn.nodes[:node_18].indices)

                    if backend_name == "DSLBackend"
                        save_output(tn, :node_18, "node_18")
                    end

                    backend_funcs[:execute]()

                    result = load_tensor_data(tn, :node_18)

                    size(result)[idx] == 2
                finally
                    backend_funcs[:finalise]()
                end
            end
        end

        @testset "Test tensor network contraction with MPS, full vector for $backend_name" begin
            @test begin
                try
                    n = 5
                    circ = create_ghz_preparation_circuit(n)
                    tn = convert_qiskit_circ_to_network(circ,
                                                        backend_funcs[:init](),
                                                        decompose=true,
                                                        transpile=true)
                    add_input!(tn, "0"^n)
                    mps_nodes = contract_mps_tensor_network_circuit!(tn)
                    calculate_mps_amplitudes!(tn, mps_nodes)
                    backend_funcs[:execute]()
                    ref_output = zeros(ComplexF64, 2^n)
                    ref_output[[1, end]] .= 1/sqrt(2)
                    load_tensor_data(tn, :result) ≈ ref_output
                finally
                    backend_funcs[:finalise]()
                end
            end
        end

        @testset "Test tensor network contraction with MPS, select amplitudes for $backend_name" begin
            @test begin
                try
                    n = 5
                    circ = create_ghz_preparation_circuit(n)
                    tn = convert_qiskit_circ_to_network(circ,
                                                        backend_funcs[:init](),
                                                        decompose=true,
                                                        transpile=true)
                    add_input!(tn, "0"^n)
                    mps_nodes = contract_mps_tensor_network_circuit!(tn)
                    backend_funcs[:execute]()
                    mps_state = MPSState(tn, mps_nodes)
                    size(mps_state) == Tuple(ones(4) .* 2)
                    length(mps_state) == 2 << n
                    mps_state["11111"] ≈ mps_state["00000"] ≈ 1/sqrt(2)
                    mps_state["10101"] ≈ 0
                finally
                    backend_funcs[:finalise]()
                end
            end
        end

        @testset "Test decopmose with threshold option for backend $backend_name" begin
            @test begin
                try
                    # create distribution to sample from to get a random unitary matrix
                    dist = Haar(1)
                    d = 2
                    random_gate_data = reshape(rand(dist, d^2), (d, d, d, d))

                    tn = TensorNetworkCircuit(2, backend_funcs[:init]())
                    add_gate!(tn, random_gate_data, [1, 2])

                    threshold = 0.5
                    # decompose gate between qubits with threshold of 0.5
                    new_nodes = decompose_tensor!(tn,
                                                :node_1,
                                                [:index_1, :index_3],
                                                [:index_2, :index_4],
                                                threshold=threshold)
                    save_output(tn, new_nodes[1], String(new_nodes[1]))

                    backend_funcs[:execute]()

                    # decompose gate data manually and check threshold was applied
                    F = svd(reshape(permutedims(random_gate_data, (1, 3, 2, 4)), (d^2, d^2)))

                    S_norm = sqrt(sum(F.S .^ 2))
                    expected_dim = sum(F.S ./ S_norm .> threshold)
                    virtual_bonds = virtualedges(tn, new_nodes[1])
                    bond_idx = findfirst(x -> x == virtual_bonds[1], tn.nodes[new_nodes[1]].indices)
                    size(load_tensor_data(tn, new_nodes[1]))[bond_idx] == expected_dim
                finally
                    backend_funcs[:finalise]()
                end
            end

            @testset "Test decopmose with max_rank option for backend $backend_name" begin
                @test begin
                    try
                        # create distribution to sample from to get a random unitary matrix
                        dist = Haar(1)
                        d = 2
                        random_gate_data = reshape(rand(dist, d^2), (d, d, d, d))

                        tn = TensorNetworkCircuit(2, backend_funcs[:init]())
                        add_gate!(tn, random_gate_data, [1, 2])

                        max_rank = 1
                        # decompose gate between qubits with threshold of 1.0
                        new_nodes = decompose_tensor!(tn,
                                                    :node_1,
                                                    [:index_1, :index_3],
                                                    [:index_2, :index_4],
                                                    max_rank=max_rank)
                        save_output(tn, new_nodes[1], String(new_nodes[1]))

                        backend_funcs[:execute]()

                        virtual_bonds = virtualedges(tn, new_nodes[1])
                        bond_idx = findfirst(x -> x == virtual_bonds[1], tn.nodes[new_nodes[1]].indices)
                        size(load_tensor_data(tn, new_nodes[1]))[bond_idx] == max_rank
                    finally
                        backend_funcs[:finalise]()
                    end
                end
            end

            @testset "Test tensor slicing backend $backend_name" begin
                @test begin
                    try
                        n = 4
                        circ = create_simple_preparation_circuit(n, 2).compose(create_qft_circuit(n))

                        # contract the regular way
                        tn = convert_qiskit_circ_to_network(circ,
                                                            backend_funcs[:init](),
                                                            decompose=true,
                                                            transpile=true)
                        add_input!(tn, "0"^n)
                        full_wavefunction_contraction!(tn, "vector")
                        backend_funcs[:execute]()
                        wf = load_tensor_data(tn, :result)

                        # now decompose to 4 partitions and contract each partition separately
                        N = 4
                        wfs = []
                        for p in 1:N
                            tn = convert_qiskit_circ_to_network(circ,
                                                                backend_funcs[:init](),
                                                                decompose=true,
                                                                transpile=true)
                            add_input!(tn, "0"^n)
                            bond_labels, bond_values = partition_network_on_virtual_bonds(tn, N, p)
                            slice_tensor_network(tn, bond_labels, bond_values)
                            full_wavefunction_contraction!(tn, "vector")
                            backend_funcs[:execute]()
                            push!(wfs, load_tensor_data(tn, :result))
                        end
                        sum(wfs) ≈ wf
                    finally
                        backend_funcs[:finalise]()
                    end
                end
            end
        end
    end
end
end