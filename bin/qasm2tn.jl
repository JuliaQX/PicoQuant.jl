#=
Script to convert QASM input to a tensor network graph file
=#

using PicoQuant
using ArgParse

"""
    function parse_commandline()

Parse command line options and return argument dictionary
"""
function parse_commandline(ARGS)
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
                action = :store_true
            "--decompose"
                help = "Decompose two qubit gates"
                action = :store_true
            "--transpile"
                help = "Transpile circuit so gates only operate between neighbouring gates"
                arg_type = Bool
                default = false
        end
        return parse_args(ARGS,s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    qasm_filename = parsed_args["qasm"]

    if parsed_args["output"] == ""
        filename_base = splitext(qasm_filename)[1]
    else
        filename_base = splitext(parsed_args["output"])[1]
    end

    output_filename = "$(filename_base).json"
    tl_filename = "$(filename_base).tl"
    data_filename = "$(filename_base).h5"

    circuit = load_qasm_as_circuit_from_file(qasm_filename)

    tng = convert_qiskit_circ_to_network(circuit, DSLBackend(tl_filename, data_filename),
                                         decompose=parsed_args["decompose"],
                                         transpile=parsed_args["transpile"])

    open(output_filename, "w") do io
        write(io, to_json(tng, parsed_args["indent"] ? 2 : 0))
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
