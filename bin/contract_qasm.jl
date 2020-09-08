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
                help = "Output file to write the contracted network structure to"
                arg_type = String
                required = true
            "--output_data", "-d"
                help = "Output file to write the contracted network data to"
                arg_type = String
                required = true
            "--decompose"
                help = "Decompose two qubit gates"
                action = :store_true
            "--transpile"
                help = "Transpile circuit so gates only operate between neighbouring gates"
                action = :store_true
            "--method", "-m"
                help = "The contraciton method to use (options, random, inorder, fullwf and mps, default fullwf)"
                arg_type = String
                default = "fullwf"
            "--indent"
                help = "Indent to use in output json file"
                action = :store_true
            "--input_config", "-c"
                help = "The starting configuration as a string of 0, 1, +, -. Default is all zeros"
                arg_type = String
                default = ""
            "--work_dir", "-w"
                help = "Folder to keep temporary files, if non given will use current folder"
                arg_type = String
                default = ""

        end
        return parse_args(ARGS, s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    work_dir = parsed_args["work_dir"] == "" ? "." : parsed_args["work_dir"]
    isdir(work_dir) || mkdir(work_dir)

    # prepend work files with work folder path
    qasm_filename = parsed_args["qasm"]
    filename_base = splitext(qasm_filename)[1]
    tl_filename = joinpath(work_dir, "$(filename_base).tl")
    data_filename = joinpath(work_dir, "$(filename_base).h5")

    # require a backend to save tensor data to
    DSLBackend(tl_filename, data_filename)
    circuit = load_qasm_as_circuit_from_file(qasm_filename)
    tng = convert_qiskit_circ_to_network(circuit,
                                         decompose=parsed_args["decompose"],
                                         transpile=parsed_args["transpile"])

    if parsed_args["input_config"] == ""
        add_input!(tng, "0"^tng.number_qubits)
    else
        @assert length(parsed_args["input_config"]) == tng.number_qubits
        add_input!(tng, parsed_args["input_config"])
    end

    method = parsed_args["method"]
    if method == "fullwf"
        full_wavefunction_contraction!(tng)
    elseif method == "mps"
        contract_mps_tensor_network_circuit!(tng)
    elseif method == "inorder"
        contract_network!(tng, inorder_contraction_plan(tng))
    elseif method == "random"
        contract_network!(tng, random_contraction_plan(tng))
    end

    execute_dsl_file(tl_filename, data_filename, parsed_args["output_data"])

    open(parsed_args["output"], "w") do io
        write(io, to_json(tng, parsed_args["indent"] ? 2 : 0))
    end
end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
