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
            "--dsl", "-f"
                help = "Name of the dsl file to load"
                required = true
            "--dsl_data", "-d"
                help = "H5 file containing the tensor data"
                required = true
            "--output", "-o"
                help = "H5 file to output data to"
                arg_type = String
                required = true
        end
        return parse_args(ARGS, s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    dsl_filename = parsed_args["dsl"]
    data_filename = parsed_args["dsl_data"]
    output_filename = parsed_args["output"]

    DSLBackend(dsl_filename, data_filename, output_filename)

    execute_dsl_file(dsl_filename, data_filename, output_filename)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
