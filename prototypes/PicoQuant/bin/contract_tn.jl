#=
Script to contract a given tensor network with a given contraction plan
=#

using PicoQuant
using ArgParse
using JSON

"""
    function parse_commandline()

Parse command line options and return argument dictionary
"""
function parse_commandline(ARGS)
    s = ArgParseSettings()
        @add_arg_table! s begin
            "--tng", "-t"
                help = "Tensor network json file to load"
                required = true
            "--plan", "-p"
                help = "Contraction plan json file to load"
                required = true
            "--output", "-o"
                help = "Output file to write contracted network to"
                arg_type = String
                default = ""
            "--indent"
                help = "Indent to use in output json file"
                arg_type = Int
                default = 0
        end
        return parse_args(ARGS, s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    tng_filename = parsed_args["tng"]
    plan_filename = parsed_args["plan"]

    tng = nothing; plan = nothing

    # Load the network from file
    open(tng_filename, "r") do io
        tng = network_from_json(read(io, String))
    end

    # Load the contraction plan from file
    open(plan_filename, "r") do io
        plan = contraction_plan_from_json(read(io, String))
    end

    contract_network!(tng, plan)

    if parsed_args["output"] == ""
        filename = "$(splitext(tng_filename)[1])_contracted.json"
    else
        filename = parsed_args["output"]
    end

    # Write the contracted network to a file
    open(filename, "w") do io
        write(io, to_json(tng, parsed_args["indent"]))
    end
end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
