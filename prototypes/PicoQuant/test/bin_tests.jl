# @testset "Test executable files in PicoQuant/bin" begin
#     gen_qft_circuit_executed = false
#     qasm2tn_executed = false
#     tn2plan_executed = false
#     contract_tn_executed = false
#     contract_qasm_executed = false
#
#     n = 4 # number of qubits for qft circuit
#
#     # Try executing the scripts
#     try
#         test_ARGS = ["--number-qubits", "$n"]
#         include("../bin/gen_qft_circuit.jl")
#         main(test_ARGS)
#         gen_qft_circuit_executed = isfile("qft_$n.qasm")
#
#         test_ARGS = ["--qasm", "qft_$n.qasm"]
#         include("../bin/qasm2tn.jl")
#         main(test_ARGS)
#         qasm2tn_executed = isfile("qft_$n.json")
#
#         test_ARGS = ["--tng", "qft_$n.json"]
#         include("../bin/tn2plan.jl")
#         main(test_ARGS)
#         tn2plan_executed = isfile("qft_$(n)_plan.json")
#
#         test_ARGS = ["--tng", "qft_$n.json", "--plan", "qft_$(n)_plan.json"]
#         include("../bin/contract_tn.jl")
#         main(test_ARGS)
#         contract_tn_executed = isfile("qft_$(n)_contracted.json")
#
#         test_ARGS = ["--qasm", "qft_$n.qasm", "--output", "output.json"]
#         include("../bin/contract_qasm.jl")
#         main(test_ARGS)
#         contract_qasm_executed = isfile("output.json")
#
#     finally
#         # clean up any created files
#         if isfile("qft_$n.qasm")
#             rm("qft_$n.qasm")
#         end
#
#         if isfile("qft_$n.json")
#             rm("qft_$n.json")
#         end
#
#         if isfile("qft_$(n)_plan.json")
#             rm("qft_$(n)_plan.json")
#         end
#
#         if isfile("qft_$(n)_contracted.json")
#             rm("qft_$(n)_contracted.json")
#         end
#
#         if isfile("output.json")
#             rm("output.json")
#         end
#     end
#
#     # Test if scripts executed
#     @test gen_qft_circuit_executed == true
#     @test qasm2tn_executed == true
#     @test tn2plan_executed == true
#     @test contract_tn_executed == true
#     @test contract_qasm_executed == true
# end
