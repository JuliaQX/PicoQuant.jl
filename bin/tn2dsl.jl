#=
Script a tensornetwork graph description to a contraction plan
=#

using PicoQuant
using ArgParse
using JSON
using HDF5

"""
    parse_commandline()

Parse command line options and return argument dictionary
"""
function parse_commandline(ARGS)
    s = ArgParseSettings()
        @add_arg_table! s begin
            "--tng", "-t"
                help = "Tensor network circuit json file to load"
                required = true
            "--tng_data", "-d"
                help = "Tensor network circuit data file to load"
                required = true
            "--dsl"
                help = "Output prefix to write contraction DSL"
                arg_type = String
                required = true
            "--output_tng", "-o"
                help = "Path to save the output tensor network graph to"
                arg_type = String
                default = ""
            "--method", "-m"
                help = "The contraciton method to use (options, random, inorder, fullwf and mps, default fullwf)"
                arg_type = String
                default = "fullwf"
            "--input_config", "-c"
                help = "The starting configuration as a string of 0, 1, +, -. Default is all zeros"
                arg_type = String
                default = ""
        end
        return parse_args(ARGS, s)
end

function main(ARGS)
    parsed_args = parse_commandline(ARGS)

    tng_filename = parsed_args["tng"]
    tng_data_filename = parsed_args["tng_data"]

    tn_json = open(tng_filename, "r") do io
        read(io, String)
    end
    
    dsl_prefix = parsed_args["dsl"]
    dsl_output = "$(dsl_prefix).tl"
    data_output = "$(dsl_prefix).h5"

    tng = network_from_json(tn_json, DSLBackend(dsl_output, data_output, "", true))
    
    # add tensors to DSL backend
    h5open(tng_data_filename, "r") do file        
        for node_label in keys(tng.nodes)
            node_label_str = String(node_label)
            data = read(file, node_label_str)
            save_tensor_data(tng, node_label, data)
        end
    end

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

    open(parsed_args["output_tng"], "w") do io
        write(io, to_json(tng, 2))
    end
end



if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
