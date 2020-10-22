#=
Script to generate QASM for Quantum Fourier Transform circuits
=#

using PyCall
using ArgParse
using PicoQuant

"""
    parse_commandline()

Parse command line options and return argument dictionary
"""
function parse_commandline(ARGS)
    s = ArgParseSettings()
        @add_arg_table! s begin
            "--number-qubits", "-n"
                help = "Size of circuit"
                arg_type = Int
                required = true
            "--output", "-o"
                help = "Output file to write tensor network graph to"
                arg_type = String
                default = ""
            "--latex-circuit"
                help = "Generate latex for circuit diagram"
                action = :store_true
            "--text-circuit"
                help = "Generate ASCII for circuit diagram"
                action = :store_true
        end
        return parse_args(ARGS, s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    qubits = parsed_args["number-qubits"]

    circuit = create_qft_circuit(qubits)

    if parsed_args["output"] == ""
        filename = "qft_$(qubits).qasm"
    else
        filename = parsed_args["output"]
    end

    open(filename, "w") do io
        write(io, circuit.qasm())
    end

    if parsed_args["latex-circuit"]
        latex_filename = "$(splitext(filename)[1]).tex"
        circuit.draw(output="latex_source", filename=latex_filename)
    end

    if parsed_args["text-circuit"]
        text_filename = "$(splitext(filename)[1]).txt"
        circuit.draw(output="text", filename=text_filename)
    end

end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
