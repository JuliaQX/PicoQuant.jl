#=
Script to convert QASM input to a tensor network graph file
=#

using PicoQuant
using ArgParse

"""
    function parse_commandline()

Parse command line options and return argument dictionary
"""
function parse_commandline()
    s = ArgParseSettings()
        @add_arg_table! s begin
            "--qasm", "-q"
                help = "QASM file to load"
                required = true
            "--output", "-o"
                help = "Output file to write tensor network graph to"
                arg_type = String
                default = ""
            "--indent"
                help = "Indent to use in output json file"
                arg_type = Int
                default = 0
        end
        return parse_args(s)
end


function main()
    parsed_args = parse_commandline()

    qasm_filename = parsed_args["qasm"]
    circuit = load_qasm_as_circuit_from_file(qasm_filename)

    tng = convert_to_tensor_network_graph(circuit)

    if parsed_args["output"] == ""
        filename = "$(splitext(qasm_filename)[1]).json"
    else
        filename = parsed_args["output"]
    end
    open(filename, "w") do io
        write(io, to_json(tng, parsed_args["indent"]))
    end
end

main()
