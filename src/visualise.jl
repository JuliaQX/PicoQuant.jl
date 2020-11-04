import MetaGraphs
import LightGraphs

using PicoQuant
using Statistics
using Logging
try
    using GraphPlot
    using Colors
catch
    @warn("GraphPlot and/or Colors not present")
end

export plot
export plot_contraction_tree, create_contraction_tree

"""
    function create_graph(tng::TensorNetworkCircuit)

Create a light graph representation of the tensor network circuit
"""
function create_graph(tng::TensorNetworkCircuit)
    # Find starting nodes and edges
    # Criteria
    # 1. Edges with nothing as source for case with no input nodes
    input_edges = [k for (k, v) in pairs(tng.edges) if v.src === nothing]
    # 2. Nodes with only outgoing edges where input nodes or contracted nodes
    source_nodes = [x for x in keys(tng.nodes) if length(inedges(tng, x)) == 0]

    # setup data structures to track vertex indices as graph is constructed
    vertex_map = Dict{Symbol, Int64}() # for tracking the index of nodes
    index_source_vertex = Dict{Symbol, Int64}()

    # Create graph and add initial nodes
    g = MetaGraphs.MetaGraph()
    layer = 0.
    # For each open input edge, we add a vertex
    for edge in input_edges
        qubit = tng.edges[edge].qubit
        MetaGraphs.add_vertex!(g,
                               Dict(:label => "input_$(qubit)",
                               :locx => layer,
                               :locy => qubit))
        index_source_vertex[edge] = LightGraphs.nv(g)
    end

    # now for node we can connect to
    for node in source_nodes
        qubits = [tng.edges[x].qubit for x in outedges(tng, node)]
        locy = mean(qubits)
        MetaGraphs.add_vertex!(g,
                               Dict(:label => String(node),
                               :locx => layer,
                               :locy => locy))
        vertex_map[node] = MetaGraphs.nv(g)
        for edge in outedges(tng, node)
            Base.push!(input_edges, edge)
            index_source_vertex[edge] = MetaGraphs.nv(g)
        end
        virtualbonds = [x for x in tng.nodes[node].indices if tng.edges[x].virtual]
        for bond in virtualbonds
            get_vertex = x -> haskey(vertex_map, x) ? vertex_map[x] : nothing
            bs = get_vertex(tng.edges[bond].src)
            bd = get_vertex(tng.edges[bond].dst)
            if bs !== nothing && bd !== nothing
                MetaGraphs.add_edge!(g, bs, bd)
                MetaGraphs.set_prop!(g, bs, bd, :label, String(bond))
            end
        end
    end

    # incremement layer index and iteratively construct the rest of graph
    layer += 1.
    # next find potentially next nodes and then iterate through
    # find nodes that are reached by current indices
    connecting_nodes = [tng.edges[x].dst for x in input_edges
                        if tng.edges[x].dst != nothing]

    while length(connecting_nodes) > 0
        for node in connecting_nodes
            incoming = inedges(tng, node)
            if length(setdiff(incoming, input_edges)) == 0
                locy = mean([tng.edges[x].qubit for x in incoming])
                MetaGraphs.add_vertex!(g, Dict(:label => String(node),
                                      :locx => layer, :locy => locy))
                vertex_idx = MetaGraphs.nv(g)
                vertex_map[node] = vertex_idx
                for edge in incoming
                    MetaGraphs.add_edge!(g, index_source_vertex[edge], vertex_idx)
                    MetaGraphs.set_prop!(g, index_source_vertex[edge], vertex_idx,
                                         :label, String(edge))
                end
                input_edges = setdiff(input_edges, incoming)
                outgoing = outedges(tng, node)
                for edge in outgoing
                    index_source_vertex[edge] = vertex_idx
                end
                input_edges = union(input_edges, outedges(tng, node))
                # add virtual bonds
                virtual = [x for x in tng.nodes[node].indices
                           if tng.edges[x].virtual]
                for bond in virtual
                    get_vertex = x -> haskey(vertex_map, x) ? vertex_map[x] : nothing
                    bs = get_vertex(tng.edges[bond].src)
                    bd = get_vertex(tng.edges[bond].dst)
                    if bs != nothing && bd != nothing
                        MetaGraphs.add_edge!(g, bs, bd)
                        MetaGraphs.set_prop!(g, bs, bd, :label, String(bond))
                    end
                end
                layer += 1.
            end
        end
        connecting_nodes = [tng.edges[x].dst for x in input_edges if tng.edges[x].dst != nothing]
    end
    g
end

"""
    function plot(tng::TensorNetworkCircuit; showlabels::Bool=false)

Function for creating a graph of a tensor network graph
"""
function plot(tng::TensorNetworkCircuit; showlabels::Bool=false)
    g = create_graph(tng)
    locx = convert(Vector{Real}, [MetaGraphs.get_prop(g, v, :locx)
                                  for v in MetaGraphs.vertices(g)])
    locy = convert(Vector{Real}, [MetaGraphs.get_prop(g, v, :locy)
                                  for v in MetaGraphs.vertices(g)])
    if showlabels
        gplot(g,
              locx,
              locy,
              edgelabel=[MetaGraphs.get_prop(g, e, :label)
                         for e in MetaGraphs.edges(g)],
              nodelabel=[MetaGraphs.get_prop(g, v, :label)
                         for v in MetaGraphs.vertices(g)])
    else
        gplot(g,
              locx,
              locy)
    end
end


# *************************************************************************** #
#                   Functions to visualise contraction plans
# *************************************************************************** #

"""
    function create_contraction_tree(network::TensorNetworkCircuit,
                                     plan::Array{Array{Symbol, 1}, 1})

This function converts a contraction plan (an array of node symbol pairs) to
a contraction tree (a MetaGraph where nodes represent tensors created during
the contraction).
"""
function create_contraction_tree(network::TensorNetworkCircuit,
                                 plan::Array{Array{Symbol, 1}, 1})

    # Convert the contraction plan to an array of integer pairs.
    edges = [[parse(Int, string(pair[1])[6:end]),
              parse(Int, string(pair[2])[6:end])]
             for pair in plan]

    # Create a metagraph structure to represent the contraction tree.
    graph = LightGraphs.Graph(length(network.nodes))
    graph = MetaGraph(graph)

    # Each node of the tree should have the uncontracted indices of the tensor
    # it represents, and the cost of the contraction that created that tensor,
    # as properties. Leaves of the tree (the tensors in the uncontracted graph)
    # should be initialised with zero cost.
    for (vertex, node) in enumerate(values(network.nodes))
        props = Dict(:cost=>0, :indices=>node.indices)
        set_props!(graph, vertex, props)
    end

    # For each edge in the contraction plan, add a new node to the tree,
    # representing the tensor created by contracting that edge, connect it to
    # the tree and give it the appropriate properties.
    for edge in edges
        LightGraphs.add_vertex!(graph)
        LightGraphs.add_edge!(graph, edge[1], LightGraphs.nv(graph))
        LightGraphs.add_edge!(graph, edge[2], LightGraphs.nv(graph))

        A_indices = get_prop(graph, edge[1], :indices)
        B_indices = get_prop(graph, edge[2], :indices)
        common_indices = intersect(A_indices, B_indices)
        remaining_indices_A = setdiff(A_indices, common_indices)
        remaining_indices_B = setdiff(B_indices, common_indices)
        remaining_indices = union(remaining_indices_A, remaining_indices_B)

        props = Dict(:cost=>length(common_indices), :indices=>remaining_indices)
        set_props!(graph, LightGraphs.nv(graph), props)
    end

    graph
end

"""
    function plot_contraction_tree(contraction_tree::MetaGraph{Int64,Float64})

Plotting function to visualise a contraction tree produced by the function
'create_contraction_tree'.
"""
function plot_contraction_tree(contraction_tree::MetaGraph{Int64,Float64})
    # TODO node sizes are too small for large networks
    node_costs = [get_prop(contraction_tree, v, :cost)
                  for v in LightGraphs.vertices(contraction_tree)]

    nodesize = [log(c+1)+1 for c in node_costs]

    num_nodes = sum(nodesize.==1)

    colors = colormap("blues", maximum(node_costs)+1)
    nodefillc = [c+1 for c in node_costs]
    nodefillc = colors[nodefillc]

    layout=(args...)->spring_layout(args...; C=0.5)

    gplot(contraction_tree, layout=layout,
          nodesize=nodesize, nodefillc=nodefillc)
end
