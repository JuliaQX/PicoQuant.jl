import LightGraphs.AbstractGraph
import LightGraphs; lg = LightGraphs
using MetaGraphs

export QuickBB_contraction!, QuickBB_contraction_plan

#= TODO: * Make the QuickBB function 'anytime'. (returns after specified time)
         * Currently, only 'graph reduction' improvement is implemented. Gogate
           mentions two more including propagtion and pruning rules.
         * Enforce assumption that input graph G is connected. =#

"""
    function QuickBB(G::TensorNetworkCircuit)

Use a contraction plan found by the QuickBB method to contract the given
TensorNetworkCircuit.
"""
function QuickBB_contraction!(tn::TensorNetworkCircuit,
                              output_shape::Union{String, Array{<:Integer, 1}}="")
    contraction_plan = QuickBB_contraction_plan(tn)
    contraction_plan = Array{Symbol, 1}(contraction_plan)
    contract_network!(tn, contraction_plan, output_shape)
end

"""
    function QuickBB(G::TensorNetworkCircuit)

Use QuickBB method to find a contraction plan for a given TensorNetworkCircuit
"""
function QuickBB_contraction_plan(tn::TensorNetworkCircuit)
    # Create the line graph of the graph underlying the network tn.
    line_graph = MetaGraph()
    vertex_map = Dict{Symbol, Int}()
    for (index, edge) in tn.edges
        (edge.src==nothing || edge.dst==nothing) && continue # Skip open indices
        lg.add_vertex!(line_graph)
        set_prop!(line_graph, lg.nv(line_graph), :ind, index)
        vertex_map[index] = lg.nv(line_graph)
    end

    # Connect the vertices of the line graph.
    for v in lg.vertices(line_graph)
        index = get_prop(line_graph, v, :ind)# Symbol("index_$(v)")
        edge = tn.edges[index]
        node_A, node_B = edge.src, edge.dst
        neighbors = union(tn.nodes[node_A].indices, tn.nodes[node_B].indices)
        neighbors = setdiff(neighbors, [index])
        for neighbor in neighbors
            if neighbor in keys(vertex_map) # if neighbor isn't an open index
                lg.add_edge!(line_graph, v, vertex_map[neighbor])
            end
        end
    end

    # Use QuickBB to find perfect elimination ordering for the line graph.
    ub, ordering = QuickBB(line_graph)

    # Convert the elimination ordering to contraction plan for G.
    contraction_plan = []
    for vertex in ordering
        ind = get_prop(line_graph, vertex, :ind)
        append!(contraction_plan, [Symbol("index_$(ind)")])
        lg.rem_vertex!(line_graph, vertex)
    end
    contraction_plan
end



"""
    function QuickBB(G::AbstractGraph)

Implements the Treewidth Branch and Bound algorithm as described in:
https://arxiv.org/abs/1207.4109
"""
function QuickBB(G::AbstractGraph)
    # Initialize state and bounds.
    x = []; g = 0; h = minor_min_width(G); f = h
    ub, ordering = min_fill_ub(G)

    if f < ub
        BB(G, x)
    end
    return ub, ordering
end

# Sub-function which recursively searches the space of perfect elimination
# orderings for G.
function BB(G::AbstractGraph, x)
    if lg.nv(G) < 2
        if f < ub
            ub = f
            ordering = x
        end
    else
        for v in lg.vertices(G)
            Gˢ = copy(G); xˢ = [x; v]
            eliminate!(Gˢ, v)
            g = max(g, lg.degree(G, v))
            h = minor_min_width(Gˢ)
            f = max(g, h)

            # Graph reduction
            reduce_graph(G, x)

            f < ub && BB(Gˢ, xˢ)
        end
    end
end

"""
function to remove vertices from G according to the
simplicial-vertex-rule and the almost-simplicial-vertex-rule.
"""
function reduce_graph(G::AbstractGraph, x)
    while true
        cliqueness_map = Dict(v => cliqueness(G,v) for v in lg.vertices(G))
        v = get_vertex_that_can_be_eliminated(cliqueness_map)
        if v === nothing
            return
        end
        g = max(g, lg.degree(G, v))
        f = max(f, g)
        append!(x, v); eliminate!(G)
    end
end

# Function to find a vertex which is simplicial or almost simplicial.
function get_vertex_that_can_be_eliminated(cliqueness_map)
    for (vi, cli_vi) in pairs(cliqueness_map)
        if cli_vi == 0 || (cli_vi == 1 && lg.degree(G, vi) < h)
            return vi
        end
    end
    return nothing
end


# *************************************************************************** #
#                    Utility functions for QuickBB method
# *************************************************************************** #

"""
    function min_fill_ub(G::AbstractGraph)

Returns the upper bound on the tree width of G found using the min-fill
heuristic.
"""
function min_fill_ub(G::AbstractGraph)
    G = copy(G)
    max_degree = 0; ordering = []

    while lg.nv(G) > 0
        # TODO: might be able to speed this up by noting which edges the
        # eliminate function adds to the graph and then increment the
        # relevent dictionary values.
        cliqueness_map = Dict(v => cliqueness(G,v) for v in lg.vertices(G))
        v = argmin(cliqueness_map)

        max_degree = max(max_degree, lg.degree(G, v))
        append!(ordering, v)
        eliminate!(G,v)
    end

    max_degree, ordering
end



"""
    function minor_min_width(G::AbstractGraph)

Returns the lower bound on the tree width of G found by minor-min-width
algorithm.
"""
function minor_min_width(G::AbstractGraph)
    # Assuming G is connected
    # TODO: This function errors in G is not connected.
    G = copy(G); lb = 0
    while lg.nv(G) > 1
        degree_map = Dict(lg.vertices(G) .=> lg.degree(G))
        min_degree, v = findmin(degree_map)

        degree_mapᵥ = Dict(u => degree_map[u] for u in lg.all_neighbors(G, v))
        u = argmin(degree_mapᵥ)

        lb = max(lb, min_degree)

        contract_vertices!(G, v, u)
    end
    lb
end



"""
    function cliqueness(G::AbstractGraph, v::Integer)

Return the number of edges that need to be added to G in order to make the
neighborhood of v simplicial.
"""
function cliqueness(G::AbstractGraph, v::Integer)
    neighborhood = lg.all_neighbors(G, v)
    count = 0
    for (i, vi) in enumerate(neighborhood)
        for ui in neighborhood[i+1:end]
            if !lg.has_edge(G, ui, vi)
                count += 1
            end
        end
    end
    count
end



"""
    function eliminate!(G::AbstractGraph, v::Integer)

Connect all the neighbors of v together before removing v from G.
"""
function eliminate!(G::AbstractGraph, v::Integer)
    neighborhood_v = lg.all_neighbors(G, v)
    for (i, vi) in enumerate(neighborhood_v)
        for ui in neighborhood_v[i+1:end]
            lg.add_edge!(G, vi, ui)
        end
    end
    lg.rem_vertex!(G, v)
end



"""
    function contract_vertices!(G::AbstractGraph, u::Integer, v::Integer)

Replaces vertices u and v with a new vertex and connects it all the neighbors
of u and v.
"""
function contract_vertices!(G::AbstractGraph, u::Integer, v::Integer)
    # Add a new vertex to replace u and v.
    lg.add_vertex!(G)

    # Connect the new vertex to the neighbors of u and v.
    for vertex in [lg.all_neighbors(G, v); lg.all_neighbors(G, u)]
        if !(vertex == v || vertex == u)
            lg.add_edge!(G, vertex, length(lg.vertices(G)))
        end
    end

    # Remove u and v from G.
    if u > v
        lg.rem_vertex!(G, u); lg.rem_vertex!(G, v)
    else
        lg.rem_vertex!(G, v); lg.rem_vertex!(G, u)
    end
end
