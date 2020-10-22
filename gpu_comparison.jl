using CUDA
using PicoQuant

let
    r = 4
    c = 4
    n = r * c
    # circ = create_ghz_preparation_circuit(n)
    circ = create_RQC(r, c, 32)

    function sim_circ_mps_gpu(circ, profile=false)
        backend = InteractiveBackend{CuArray{ComplexF64}}()
       	tng = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true, backend)
        add_input!(tng, "0"^circ.n_qubits)
        CUDA.synchronize()        
        if !profile
            CUDA.@time output_node = contract_mps_tensor_network_circuit!(tng) 
        else
            CUDA.@profile output_node = contract_mps_tensor_network_circuit!(tng) 
        end
    end

    function sim_circ_mps_cpu(circ)
        backend = InteractiveBackend{Array{ComplexF32}}()
       	tng = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true, backend)
        add_input!(tng, "0"^circ.n_qubits)        
        
        @time output_node = contract_mps_tensor_network_circuit!(tng) 
    end

    println("Run CPU version")
    # sim_circ_mps_cpu(circ)
    sim_circ_mps_cpu(circ)

    CUDA.allowscalar(false)

    println("Run GPU version")
    # sim_circ_mps_gpu(circ)
    sim_circ_mps_gpu(circ)

    sim_circ_mps_gpu(circ, true)
end



