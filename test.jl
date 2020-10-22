using CUDA
using PicoQuant
using BenchmarkTools

let
    r = 4
    c = 4
    n = 16
    circ = create_ghz_preparation_circuit(n)
    #circ = create_RQC(r, c, 8)
    

    function sim_circ_full(circ, gpu=false)
        backend = gpu ? InteractiveBackend{CuArray{ComplexF32}}() : InteractiveBackend{Array{ComplexF32}}()
       	tng = convert_qiskit_circ_to_network(circ, backend)
	add_input!(tng, "0"^n)
        if gpu
	    CUDA.@time output_node = full_wavefunction_contraction!(tng) 
        else
	    @time output_node = full_wavefunction_contraction!(tng) 
        end
    end

    function sim_circ_mps(circ, gpu=false)
        backend = gpu ? InteractiveBackend{CuArray{ComplexF32}}() : InteractiveBackend{Array{ComplexF32}}()
       	tng = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true, backend)
	add_input!(tng, "0"^n)
        if gpu
	    CUDA.@profile output_node = contract_mps_tensor_network_circuit!(tng) 
        else
	    @time output_node = contract_mps_tensor_network_circuit!(tng) 
        end
    end


    #sim_circ_full(circ)
    #sim_circ_full(circ)

    #sim_circ_full(circ, true)
    #sim_circ_full(circ, true)

    sim_circ_mps(circ)
    sim_circ_mps(circ)

    sim_circ_mps(circ, true)
    sim_circ_mps(circ, true)

end



