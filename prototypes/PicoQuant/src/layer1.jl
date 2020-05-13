export load_tensor!, save_tensor!, delete_tensor!, contract_tensors
export reshape_tensor, transpose_tensor, conjugate_tensor, execute_dsl_file

using TensorOperations, HDF5

# *************************************************************************** #
#                             dsl functions
# *************************************************************************** #
"""
    function load_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                          tensor_label::String,
                          tensor_data_filename::String)

Load tensor data, identified by tensor_label, from a .h5 file and store
it in the dictionary 'tensors'
"""
function load_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                      tensor_label::String,
                      tensor_data_filename::String)

    tensor = Symbol(tensor_label)
    tensors[tensor] = h5open(tensor_data_filename, "r") do file
        read(file, tensor_label)
    end
end

"""
    function save_tensor!(tensor_data_filename::String,
                          tensors::Dict{Symbol, Array{<:Number}},
                          tensor_label::String,
                          group_name::String)

Save tensor data to a .h5 file. If no group name is given for the data the
label that identifies the tensor in the dictionary 'tensors' is used.
"""
function save_tensor!(tensor_data_filename::String,
                      tensors::Dict{Symbol, Array{<:Number}},
                      tensor_label::String,
                      group_name::String="")

    # If no group name given then use the tensor label.
    if group_name == ""
        group_name = tensor_label
    end

    tensor_label = Symbol(tensor_label)
    h5write(tensor_data_filename, group_name, tensors[tensor_label])
end

"""
    function delete_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                            tensor_label::String)

Delete the specified tensor from the dictionary 'tensors'
"""
function delete_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                        tensor_label::String)

    delete!(tensors, Symbol(tensor_label))
end

"""
    function contract_tensors!(tensors::Dict{Symbol, Array{<:Number}},
                               new_label::String,
                               A_label::String, A_indices::String,
                               B_label::String, B_indices::String)

Contract the tensors A and B and save the result as a new tensor
"""
function contract_tensors(tensors_to_contract, tensor_indices)
    ncon(tensors_to_contract, tensor_indices)
end

"""
    function reshape!(tensors::Dict{Symbol, Array{<:Number}},
                      tensor_label::Symbol,
                      dims::Array{<:Integer})

Reshape a tensor
"""
function reshape_tensor(tensor, dims)
    reshape(tensor, dims...)
end

"""
    function transpose_tensor(tensor::Array{<:Number},
                              index_permutation::Array{<:Integer, 1})

Transpose a tensor by permuting the indices as specified in index_permutation.
"""
function transpose_tensor(tensor::Array{<:Number},
                   index_permutation::Array{<:Integer, 1})
    permutedims(tensor, index_permutation)
end

"""
    function conjugate_tensor(tensor::Array{<:Number})

Conjugate the elements of a tensor.
"""
function conjugate_tensor(tensor::Array{<:Number})
    conj.(tensor)
end

# *************************************************************************** #
#                Reading and executing dsl commands from file
# *************************************************************************** #

"""
    function execute_dsl_file(dsl_filename::String,
                              tensor_data_filename::String)

Contract the tensors A and B and save the result as a new tensor
"""
function execute_dsl_file(dsl_filename::String="contract_network.tl",
                          tensor_data_filename::String="tensor_data.h5",
                          output_data_filename::String="")

    # Open file to read dsl commands from.
    file = open(dsl_filename)

    # Create dictionary to hold tensors
    tensors = Dict{Symbol, Array{<:Number}}()

    # Read each line of the dsl file and execute specified functions.
    for command in eachline(file)
        # Split the line into the command name and arguments.
        command = string.(split(command))

        if command[1] == "ncon"
            C_label, A_label, A_indices, B_label, B_indices = command[2:end]

            C_label = Symbol(C_label)
            A_label = Symbol(A_label)
            B_label = Symbol(B_label)

            # Convert indices into integer array for ncon.
            A_indices = parse.(Int, split(A_indices, ","))
            B_indices = parse.(Int, split(B_indices, ","))

            # Contract A and B and save the result.
            A = tensors[A_label]; B = tensors[B_label]
            tensors[C_label] = contract_tensors((A, B), (A_indices, B_indices))

        elseif command[1] == "del"
            delete_tensor!(tensors, command[2])

        elseif command[1] == "tensor"
            load_tensor!(tensors, command[2], tensor_data_filename)

        elseif command[1] == "save"
            tensor_to_save = command[2]
            output_data_filename = command[3]
            group_name = command[4]
            save_tensor!(output_data_filename, tensors,
                         tensor_to_save, group_name)

        elseif command[1] == "reshape"
            tensor_label, dims = command[2:end]
            tensor_label = Symbol(tensor_label)
            tensor = tensors[tensor_label]

            # Convert dimensions into integer array for reshaping.
            dims = parse.(Int, split(dims, ","))

            # Reshape the tensor.
            tensors[tensor_label] = reshape_tensor(tensor, dims)
        end
    end

    close(file)
end
