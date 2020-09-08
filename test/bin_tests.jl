@testset "Test executable files in PicoQuant/bin" begin
    gen_qft_circuit_executed = false
    qasm2tn_executed = false
    tn2plan_executed = false
    contract_tn_executed = false
    contract_qasm_executed = false

    n = 4 # number of qubits for qft circuit
    test_dir = "tmptest"
    if isdir(test_dir) rm(test_dir, recursive=true) end
    mkdir("tmptest")

    # Try executing the scripts
    try
        test_ARGS = ["--number-qubits", "$n", "-o", joinpath(test_dir, "qft_$n.qasm")]
        include("../bin/gen_qft_circuit.jl")
        main(test_ARGS)
        gen_qft_circuit_executed = isfile(joinpath(test_dir, "qft_$n.qasm"))

        test_ARGS = ["--qasm", joinpath(test_dir, "qft_$n.qasm")]
        include("../bin/qasm2tn.jl")
        main(test_ARGS)
        qasm2tn_executed = isfile(joinpath(test_dir, "qft_$n.json"))  &&
                           isfile(joinpath(test_dir, "qft_$n.tl")) &&
                           isfile(joinpath(test_dir, "qft_$n.h5"))

        test_ARGS = ["--tng", joinpath(test_dir, "qft_$n.json"),
                     "--tng_data", joinpath(test_dir, "qft_$n.h5"),
                     "--dsl", joinpath(test_dir, "qft_$(n)_contract"),
                     "--output_tng", joinpath(test_dir, "qft_$(n)_contract.json")]
        include("../bin/tn2dsl.jl")
        main(test_ARGS)
        tn2plan_executed = isfile(joinpath(test_dir, "qft_$(n)_contract.tl")) &&
                           isfile(joinpath(test_dir, "qft_$(n)_contract.h5")) &&
                           isfile(joinpath(test_dir, "qft_$(n)_contract.json"))

        test_ARGS = ["--dsl", joinpath(test_dir, "qft_$(n)_contract.tl"),
                     "--dsl_data", joinpath(test_dir, "qft_$(n)_contract.h5"),
                     "--output", joinpath(test_dir, "qft_$(n)_contract_out.h5")]
        include("../bin/contract_tn.jl")
        main(test_ARGS)
        contract_tn_executed = isfile(joinpath(test_dir, "qft_$(n)_contract_out.h5"))

        test_ARGS = ["--qasm", joinpath(test_dir, "qft_$n.qasm"),
                     "--output", joinpath(test_dir, "output.json"),
                     "--output_data", joinpath(test_dir, "output.h5")]
        include("../bin/contract_qasm.jl")
        main(test_ARGS)
        contract_qasm_executed = isfile(joinpath(test_dir, "output.json")) &&
                                 isfile(joinpath(test_dir, "output.h5"))

    finally
        # clean up any created files
        for fn in ["qft_$n.qasm",
                   "qft_$n.json",
                   "qft_$n.tl",
                   "qft_$n.h5",
                   "qft_$(n)_contract.tl",
                   "qft_$(n)_contract.h5",
                   "qft_$(n)_contract.json",
                   "qft_$(n)_contract_out.h5",
                   "output.json",
                   "output.h5"]
            rm(joinpath(test_dir, fn), force=true)
        end
        rm(test_dir, recursive=true)
    end

    # Test if scripts executed
    @test gen_qft_circuit_executed == true
    @test qasm2tn_executed == true
    @test tn2plan_executed == true
    @test contract_tn_executed == true
    @test contract_qasm_executed == true
end
