module DistSlicingExample
    using PicoQuant
    using MPI

    MPI.Init()
    comm = MPI.COMM_WORLD
    number_partitions = MPI.Comm_size(comm)
    partition = MPI.Comm_rank(comm) + 1

    MPI.Barrier(comm)

    n = 4
    circ = create_simple_preparation_circuit(n, 2).compose(create_qft_circuit(n))

    if partition == 1
        # contract the regular way
        tn = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true)
        add_input!(tn, "0"^n)
        wf = full_wavefunction_contraction!(tn, "vector")
    end

    tn = convert_qiskit_circ_to_network(circ, decompose=true, transpile=true)
    add_input!(tn, "0"^n)
    bond_labels, bond_values = partition_network_on_virtual_bonds(tn,
                                                                  number_partitions,
                                                                  partition)
    slice_tensor_network(tn, bond_labels, bond_values)
    wf_part = full_wavefunction_contraction!(tn, "vector")

    MPI.Reduce!(wf_part, MPI.SUM, 0, MPI.COMM_WORLD)
    if partition == 1
        if wf ≈ wf_part
            println("Results match with overlap: ", sum(wf .* conj(wf)))
        else
            println("Result mismatch with overlap: ", sum(wf .* conj(wf)))
        end
    end
end