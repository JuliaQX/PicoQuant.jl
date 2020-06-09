#=
Script to read in a circuit from a qasm file, convert it to a tensor network
and contract it.
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
            "--qasm", "-q"
                help = "QASM file to load"
                required = true
            "--output", "-o"
                help = "Output file to write the contracted network to"
                arg_type = String
                default = ""
            "--input-config", "-i"
                help = "The input configuration to use (default is all 0's)"
                arg_type = String
                default = ""
            "--output-config", "-a"
                help = "Configuration to calculate amplitude for (default all)"
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

    # Load the circuit from the given qasm file.
    qasm_filename = parsed_args["qasm"]
    circuit = load_qasm_as_circuit_from_file(qasm_filename)
    tng = convert_qiskit_circ_to_network(circuit)

    if parsed_args["input-config"] == ""
        add_input!(tng, "0"^tng.number_qubits)
    else
        @assert length(parsed_args["input-config"]) == tng.number_qubits
        add_input!(tng, parsed_args["input-config"])
    end

    if parsed_args["output-config"] != ""
        @assert length(parsed_args["output-config"]) == tng.number_qubits
        add_output!(tng, parsed_args["output-config"])
    end

    # Get a contraction plan and contract the network.
    plan = random_contraction_plan(tng)

    contract_network!(tng, plan)

    # Write the contracted network to a json file.
    if parsed_args["output"] == ""
        filename = "$(splitext(qasm_filename)[1])_contracted.json"
    else
        filename = parsed_args["output"]
    end
    open(filename, "w") do io
        write(io, to_json(tng, parsed_args["indent"]))
    end
end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
