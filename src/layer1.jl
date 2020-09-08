export load_tensor!, save_tensor!, delete_tensor!, contract_tensors
export reshape_tensor, transpose_tensor, conjugate_tensor, execute_dsl_file
export decompose_tensor, permute_tensor

using TensorOperations, HDF5
using JSON

# *************************************************************************** #
#                             dsl functions
# *************************************************************************** #
"""
    function load_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                          tensor_label::String,
                          data_label::String,
                          tensor_data_filename::String)

Load tensor data, identified by tensor_label, from a .h5 file and store
it in the dictionary 'tensors'
"""
function load_tensor!(tensors::Dict{Symbol, Array{<:Number}},
                      tensor_label::String,
                      data_label::String,
                      tensor_data_filename::String)

    tensor = Symbol(tensor_label)
    tensors[tensor] = h5open(tensor_data_filename, "r") do file
        read(file, data_label)
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
    h5open(tensor_data_filename, "cw") do file
        # TODO: Maybe this first if clause is redundant.
        if exists(file, tensor_label)
            o_delete(file, tensor_label)
        end
        if exists(file, group_name)
            o_delete(file, group_name)
        end
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
    function contract_tensors(tensors_to_contract::Tuple{Array{<:Number}, Array{<:Number}},
                              tensor_indices::Tuple{Array{<:Integer,1},Array{<:Integer,1}})

Function to contract the tensors contained in the tuple 'tensors_to_contract'
according to the ncon indices given and return the result.
"""
function contract_tensors(tensors_to_contract::Tuple{Array{<:Number}, Array{<:Number}},
                          tensor_indices::Tuple{Array{<:Integer,1},Array{<:Integer,1}})
    ncon(tensors_to_contract, tensor_indices)
end

"""
    function reshape_tensor(tensor, dims)

Reshape a tensor
"""
function reshape_tensor(tensor::Array{<:Number},
                        dims::Union{Integer, Array{<:Integer, 1}})
    reshape(tensor, dims...)
end

"""
    function permute_tensor(tensor, dims)

Permute a tensor
"""
function permute_tensor(tensor::Array{<:Number},
                        dims::Union{Integer, Array{<:Integer, 1}})
    permutedims(tensor, dims)
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

function decompose_tensor(tensor::Array{<:Number},
                          left_positions::Array{Int, 1},
                          right_positions::Array{Int, 1};
                          threshold::AbstractFloat=1e-13,
                          max_rank::Integer=0)

    dims = size(tensor)
    left_dims = [dims[x] for x in left_positions]
    right_dims = [dims[x] for x in right_positions]

    A = permutedims(tensor, vcat(left_positions, right_positions))
    A = reshape(A, Tuple([prod(left_dims), prod(right_dims)]))

    # Use SVD here but QR could also be used
    F = svd(A)

    # find number of singular values above the threshold
    chi = sum(F.S .> threshold)
    if max_rank > 0
        chi = min(max_rank, chi)
    end
    s = sqrt.(F.S[1:chi])

    # assume that singular values and basis of U and V matrices are sorted
    # in descending order of singular value
    B = reshape(F.U[:, 1:chi] * Diagonal(s), Tuple(vcat(left_dims, [chi,])))
    C = reshape(Diagonal(s) * F.Vt[1:chi, :], Tuple(vcat([chi,], right_dims)))

    B, C
end


# *************************************************************************** #
#                Reading and executing dsl commands from file
# *************************************************************************** #

"""
    function execute_dsl_file(dsl_filename::String,
                              tensor_data_filename::String)

Function to read dsl commands from a given dsl file and execute them using
tensor data from the given tensor data file.
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
            load_tensor!(tensors, command[2], command[3], tensor_data_filename)

        elseif command[1] == "save"
            tensor_to_save = command[2]
            if output_data_filename == ""
                output_data_filename = command[3]
            end
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

        elseif command[1] == "permute"
            tensor_label, dims = command[2:end]
            tensor_label = Symbol(tensor_label)
            tensor = tensors[tensor_label]

            # Convert dimensions into integer array for reshaping.
            dims = parse.(Int, split(dims, ","))

            # Reshape the tensor.
            tensors[tensor_label] = permute_tensor(tensor, dims)

        elseif command[1] == "decompose"
            A_label, B_label, B_idxs, C_label, C_idxs = command[2:6]
            options = join(command[7:end])

            A_label = Symbol(A_label)
            B_label = Symbol(B_label)
            C_label = Symbol(C_label)

            # Convert indices into integer array for ncon.
            B_idxs = parse.(Int, split(B_idxs, ","))
            C_idxs = parse.(Int, split(C_idxs, ","))

            # Contract A and B and save the result.
            A = tensors[A_label]

            (B, C) = decompose_tensor(A,
                                        B_idxs,
                                        C_idxs;
                                        JSON.parse(options, dicttype=Dict{Symbol, Any})...)

            tensors[B_label] = B
            tensors[C_label] = C
        end
    end

    close(file)
end
