export bgreedy, bgreedy_contraction!

import StatsBase.sample
import StatsBase.weights

# *************************************************************************** #
#                         Boltzmann greedy functions
# *************************************************************************** #

"""
    function bgreedy(network::TensorNetworkCircuit, α::Real, τ::Real)

This function returns a contraction plan for a network by sequentially choosing
random pairs of nodes to contract according to a Boltzmann distribution where
expensive contractions are less likely to be chosen.
"""
function bgreedy(network::TensorNetworkCircuit, α::Real, τ::Real)

    tensors = copy(network.nodes)
    contraction_plan = Array{Array{Symbol, 1}, 1}()
    number_unique_tensors = network.counters["node"]
    time_cost = 0; space_cost = 0

    while length(tensors) > 1
        # Randomly choose two tensors A and B to contract. The resulting
        # tensor is denoted C.
        C, A_label, B_label, costs = get_random_pair(tensors, α, τ)
        number_unique_tensors += 1
        C_label = Symbol("node_$number_unique_tensors")

        # Remove A and B from tensors and add C
        # Append the contraction (A, B) to the contraction plan.
        delete!(tensors, A_label)
        delete!(tensors, B_label)
        tensors[C_label] = C
        append!(contraction_plan, [[A_label, B_label]])

        time_cost += costs[1]
        if space_cost < costs[2]
            space_cost = costs[2]
        end
    end

    contraction_plan, time_cost, space_cost
end

"""
    function bgreedy(network::TensorNetworkCircuit, α::Real, τ::Real, N::Int)

This function samples N contraction plans by calling the bgreedy function
N times and returns the best contraction plan found.
"""
function bgreedy(network::TensorNetworkCircuit, α::Real, τ::Real, N::Int)
    best_plan = []
    best_time_cost = Inf

    for n = 1:N
        plan, time_cost, space_cost = bgreedy(network, α, τ)
        if time_cost < best_time_cost
            best_time_cost = time_cost
            best_plan = plan
        end
    end

    best_plan, best_time_cost
end

"""
    function contraction_cost(A::Node, B::Node, α::Real)

This function returns the heuristic cost of contracting tensors A and B given
by Gray and Kourtis in 'arXiv:2002.01935'. The labels and dimensions for the
uncontracted indices and both the time cost (flops) and space cost (memory) of
the contraction are also returned.
"""
function contraction_cost(A::Node, B::Node, α::Real)
    contracted_indices, C_indices = sort_indices(A, B)

    A_open_dims = [A.dims[i] for i = 1:length(A.dims)
                   if A.indices[i] in C_indices]
    B_open_dims = [B.dims[i] for i = 1:length(B.dims)
                   if B.indices[i] in C_indices]
    C_dims = [A_open_dims; B_open_dims]

    size_C = prod(C_dims)
    size_A = prod(A.dims)
    size_B = prod(B.dims)

    # Note: sqrt(|A||B||C|) = |open dims| x |contracted dims| ∝ flops
    costs = [sqrt(size_A*size_B*size_C), size_C]

    # return the Gray and Kourtis heuristic cost for the given value of α.
    return size_C - α*(size_A + size_B), C_indices, C_dims, costs
end

"""
    function get_random_pair(tensors::Dict{Symbol, Node}, α::Real, τ::Real)

This function will return a pair of tensors A and B sampled from the boltzmann
distribution over the given dictionary of tensors. The cost of contracting A and
B plays the role of energy. (Cheaper contractions are more likely to by chosen.)
A Node instance for the tensor C resulting from contracting A and B is also
returned.
"""
function get_random_pair(tensors::OrderedDict{Symbol, Node}, α::Real, τ::Real)
    # Allocate memory
    N = length(tensors); N_pairs = N*(N-1)÷2
    tensors = collect(tensors)
    tensor_pairs = Array{Array{Symbol, 1}, 1}(undef, N_pairs)
    energy = Array{Real, 1}(undef, N_pairs)
    C_indices = Array{Array{Symbol, 1}, 1}(undef, N_pairs)
    C_dims = Array{Array{Int, 1}, 1}(undef, N_pairs)
    costs = Array{Array{Real, 1}, 1}(undef, N_pairs)

    # Loop over all possible pairings of tensors and compute heuristic cost
    k = 1
    for i = 1:N, j = i+1:N
        (A_label, A) = tensors[i]; (B_label, B) = tensors[j]
        tensor_pairs[k] = [A_label, B_label]
        energy[k], C_indices[k], C_dims[k], costs[k] = contraction_cost(A, B, α)
        k += 1
    end

    # Sample a pair according to the Boltzmann distribution and return
    # the result
    weights_vector = exp.(-(energy)./τ)
    k = sample(1:N_pairs, weights(weights_vector))
    A_label, B_label = tensor_pairs[k]
    C = Node(C_indices[k], C_dims[k], :intermediate_tensor)
    return C, A_label, B_label, costs[k]
end

"""
    function bgreedy_contraction!(network::TensorNetworkCircuit,
                                  α::Real, τ::Real, N::Int)

Function to contract a TensorNetworkCircuit with a contraction plan found by
the bgreedy method with the given parameters.
"""
function bgreedy_contraction!(network::TensorNetworkCircuit,
                              α::Real=1, τ::Real=1, N::Int=10,
                              output_shape::Union{String, Array{Int, 1}}="")
    contraction_plan, cost = bgreedy(network, α, τ, N)
    contract_network!(network, contraction_plan, output_shape)
end
