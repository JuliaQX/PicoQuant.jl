module SlicingExample
    using PicoQuant

    n = 4        
    circ = create_simple_preparation_circuit(n, 2).compose(create_qft_circuit(n))

    # contract the regular way
    tn = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true)
    add_input!(tn, "0"^n)
    wf = full_wavefunction_contraction!(tn, "vector")

    # now decompose to 4 partitions and contract each partition separately
    N = 4
    wfs = []
    for p in 1:N
        tn = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true)
        add_input!(tn, "0"^n)
        bond_labels, bond_values = partition_network_on_virtual_bonds(tn, N, p)
        slice_tensor_network(tn, bond_labels, bond_values)
        push!(wfs, full_wavefunction_contraction!(tn, "vector"))
    end    
    if sum(wfs) ≈ wf
        println("Results match with and without partitioning")
        print("Overlap: ")
        println(sum(wf .* conj(sum(wfs))))
    else
        println("Discrepency between results with and without partitioning")
        println("$wf")
        println("$(sum(wfs))")
    end

end