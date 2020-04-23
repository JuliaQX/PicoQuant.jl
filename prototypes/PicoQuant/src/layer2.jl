import Random.shuffle

export random_contraction_path
export contraction_path_to_json, contraction_path_from_json



function random_contraction_path(network::TensorNetworkCircuit)
    path = edges(network)
    shuffle(path)
end

# function cost_flops(network, path)
# end
#
# function cost_max_memory(network, path)
# end
#
# function find_contraction_plan(network, cost_function)
# end



"""
    function contraction_path_to_json(plan::Array)

Function to serialise the contraction plan to json format
"""
function contraction_path_to_json(path::Array{<:Array{<:Integer, 1}, 1})
    JSON.json(path)
end

"""
    function contraction_path_from_json(str::String)

Function to deserialize the contraction plan from a json string
"""
function contraction_path_from_json(str::String)
    path = convert(Array{Array{Int, 1}, 1}, JSON.parse(str))
end
