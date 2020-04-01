import Base: ==



struct ContractionPlan
    edges::Array{Array{Int64, 1}, 1}
end

export ContractionPlan, simple_contraction_plan, to_json, contraction_plan_from_json

"""
    function ==(a::ContractionPlan, b::ContractionPlan)

Compare two instances of contraction plan
"""
function ==(a::ContractionPlan, b::ContractionPlan)
    a.edges == b.edges
end

"""
    function simple_contraction_plan(tng::TensorNetworkCircuit)

Function to create simple contraction plan starting from inputs
"""
function simple_contraction_plan(tng::TensorNetworkCircuit)
    contraction_plan = [[e.src, e.dst] for e in edges(tng.graph) if get_prop(tng.graph, e.dst, :type) != "output"]
    out_edges = [[e.src, e.dst] for e in edges(tng.graph) if get_prop(tng.graph, e.dst, :type) == "output"]
    ContractionPlan(vcat(contraction_plan, out_edges))
end

"""
    function to_json(plan::Array)

Function to serialise the contraction plan to json format
"""
function to_json(plan::ContractionPlan)
    JSON.json(plan.edges)
end

"""
    function contraction_plan_from_json(str::String)

Function to deserialize the contraction plan from a json string
"""
function contraction_plan_from_json(str::String)
    edges = JSON.parse(str)
    ContractionPlan(convert(Array{Array{Int64, 1}, 1}, edges))
end
