using FFTW

@testset "Test state preparation algorithm" begin
    @test begin
        qubits = 1
        circ = create_simple_preparation_circuit(qubits, 4, 42)

        ψ = get_statevector_using_picoquant(circ,
                                            big_endian=false)
        ψ′ = get_statevector_using_qiskit(circ, big_endian=false)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ψ′) ≈ 1.
    end

    @test begin
        qubits = 3
        circ = create_simple_preparation_circuit(qubits, 4, 42)

        ψ = get_statevector_using_picoquant(circ,
                                            big_endian=false)
        ψ′ = get_statevector_using_qiskit(circ, big_endian=false)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ψ′) ≈ 1.
    end
end

@testset "Small QFT circuit test" begin
    @test begin
        n = 3

        prep_circ = create_simple_preparation_circuit(n, 3, 43)
        qft_circ = create_qft_circuit(n)

        full_circ = prep_circ.combine(qft_circ)

        qft_input = get_statevector_using_qiskit(prep_circ, big_endian=false)
        ref_output = ifft(qft_input)
        norm = x -> x ./ sqrt(sum(x .* conj.(x)))
        ref_output = norm(ref_output)

        ψ = get_statevector_using_picoquant(full_circ)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ref_output) ≈ 1.
    end
end


@testset "Larger QFT circuit test" begin
    @test begin
        n = 8

        prep_circ = create_simple_preparation_circuit(n, 3, 43)
        qft_circ = create_qft_circuit(n)

        full_circ = prep_circ.combine(qft_circ)

        qft_input = get_statevector_using_qiskit(prep_circ, big_endian=false)
        ref_output = ifft(qft_input)
        norm = x -> x ./ sqrt(sum(x .* conj.(x)))
        ref_output = norm(ref_output)

        ψ = get_statevector_using_picoquant(full_circ)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(ψ, ref_output) ≈ 1.
    end
end

@testset "Small RQC circuit test" begin
    @test begin
        n = 3; m = 3; depth = 8

        rqc = create_RQC(n, m, depth)

        qiskit_Ψ = get_statevector_using_qiskit(rqc, big_endian=false)

        picoquant_ψ = get_statevector_using_picoquant(rqc)

        overlap = (a, b) -> abs(transpose(conj.(a)) * b)

        overlap(picoquant_ψ, qiskit_Ψ) ≈ 1.
    end
end
