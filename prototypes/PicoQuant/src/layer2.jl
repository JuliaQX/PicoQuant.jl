import Random.shuffle

export random_contraction_plan
export contraction_plan_to_json, contraction_plan_from_json

function random_contraction_plan(network::TensorNetworkCircuit)
    closed_edges = [x for (x, y) in pairs(edges(network)) if y.src != nothing
                                                          && y.dst != nothing]
    shuffle(closed_edges)
end

# function cost_flops(network, plan)
# end
#
# function cost_max_memory(network, plan)
# end
#
# function find_contraction_plan(network, cost_function)
# end


"""
    function contraction_plan_to_json(plan::Array)

Function to serialise the contraction plan to json format
"""
function contraction_plan_to_json(plan::Array{Symbol})
    JSON.json(plan)
end

"""
    function contraction_plan_from_json(str::String)

Function to deserialize the contraction plan from a json string
"""
function contraction_plan_from_json(str::String)
    [Symbol(x) for x in JSON.parse(str)]
end
