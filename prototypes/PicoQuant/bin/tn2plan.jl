#=
Script a tensornetwork graph description to a contraction plan
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
                help = "Tensornetwork circuit json file to load"
                required = true
            "--output", "-o"
                help = "Output file to write contraction plan to"
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

    open(tng_filename, "r") do io
        tng = network_from_json(read(io, String))

        plan = random_contraction_plan(tng)

        if parsed_args["output"] == ""
            filename = "$(splitext(tng_filename)[1])_plan.json"
        else
            filename = parsed_args["output"]
        end
        open(filename, "w") do io
            write(io, JSON.json(plan, parsed_args["indent"]))
        end

    end
end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
