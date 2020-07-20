using PicoQuant
using LaTeXStrings
import Plots
#=
This script produces graphs showing the output distribution of a rqc converges
to the Porter-Thomas distribution as the depth of the rqc increases.
=#

# Set parameters of the rqc & number of samples to take for each value of depth.
depth = 30; n = 4; m = 4; samples = 100

# Allocate memory for entropy and inverse participation ratios.
average_entropy = zeros(depth); average_sqrd_entropy = zeros(depth)
IPR = zeros(5, depth); IPR_sqrd = zeros(5, depth)

# For each sample at each circuit depth, create a RQC, convert it to a
# TensorNetworkCircuit, contract the network to get the output state and
# calculate the entropy and inverse participation ratios.
for (i, d) in enumerate(1:depth)
    for _ = 1:samples
        rqc = create_RQC(n, m, d)

        InteractiveBackend()
        rqc_tn = convert_qiskit_circ_to_network(rqc)
        add_input!(rqc_tn, "0"^length(rqc_tn.input_qubits))

        output_state = full_wavefunction_contraction!(rqc_tn, "vector")
        output_distribution = abs.(output_state).^2

        for k = 1:5
            ipr = 1/sum(output_distribution.^(2*k))
            IPR[k, d] += ipr
            IPR_sqrd[k, d] += ipr^2
        end

        entropy = -sum(output_distribution.*log.(output_distribution))
        average_entropy[i] += entropy
        average_sqrd_entropy[i] += entropy^2
    end
end

average_entropy /= samples
average_sqrd_entropy /= samples
IPR /= samples; IPR_sqrd /= samples

# Get the error in the average entropy.
error = sqrt.(abs.(average_sqrd_entropy - average_entropy.^2))./sqrt(samples)

# Plot the average entropy of the output distribution as a function of depth.
plt = Plots.plot(1:depth, average_entropy, xlabel = "Depth", ylabel = "Entropy",
                 label = "Average entropy of output distribution",
                 yerror = error,
                 title = "Convergence to Porter-Thomas Entropy")

N = 2^(n*m)
porter_thomas_entropy = log(N) - 1 + 0.5772156649

plt = Plots.plot!(plt, 1:depth, porter_thomas_entropy*ones(depth),
                  label = "Porter-Thomas entropy")
display(plt)
Plots.savefig("entropy_convergence.png")

# Divide by Porter-Thomas IPR and get the IPR error
N = BigFloat(N)
for k = 1:5
    IPR[k, :] *= factorial(2*k)/N^(2*k - 1)
    IPR_sqrd[k, :] *= (factorial(2*k)/(N^(2*k - 1)))^2
end
IPR_error = sqrt.(abs.(IPR_sqrd - IPR.^2))./sqrt(samples)

# Plot the average inverse participation ratio of the Porter-Thomas distribution
# divided by that of the RQC output distribution as a function of depth.
plt2 = Plots.plot(xlabel = "Depth", ylabel = L"N^{-k+1}k!/IPR^k",
                  title = "Convergence to Porter-Thomas IPR",
                  include_mathjax = "cdn")
for k = 1:5
    global plt2
    plt2 = Plots.plot!(plt2, 3:depth, IPR[k, 3:end],
                       label = "k = $(2*k)", yaxis = :log, color=k)
    plt2 = Plots.plot!(plt2, 3:depth, IPR[k, 3:end]+IPR_error[k, 3:end],
                       label = :none, yaxis = :log,
                       seriestype = :scatter, markershape = :hline, color=k)
end
display(plt2)
Plots.savefig("IPR_convergence.png")
