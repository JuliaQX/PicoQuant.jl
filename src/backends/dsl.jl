import Base.push!

export DSLBackend

# *************************************************************************** #
#                               DSL Backend
# *************************************************************************** #

"
The dsl backend will get functions that act on a tensor network circuit to
write dsl commands. It also holds filenames for files containing dsl commands,
tensor data and the result of fully contracting a tensor network circuit.
"
mutable struct DSLBackend <: AbstractBackend
    dsl_filename::String
    tensor_data_filename::String
    output_data_filename::String
    metrics::Metrics

    function DSLBackend(dsl::String="contract_network.tl",
                        tensor_data="tensor_data.h5"::String,
                        output::String="", overwrite::Bool=false)
        # Create an empty dsl file if none exists or overwrite is selected
        if !isfile(dsl) || overwrite
            close(open(dsl, "w"))
        end

        # If no output filename given by the user, use the tensor data file
        # to store output
        if output==""
            output = tensor_data
        end

        # empty the file if it exists
        if !isfile(tensor_data) || overwrite
            close(h5open(tensor_data, "w"))
        end

        new(dsl, tensor_data, output, Metrics())
    end
end


"""
    function push!(dsl::DSLBackend, instruction::String)

Function to append a dsl command to the dsl script contained in a dsl backend.
"""
function push!(dsl::DSLBackend, instruction::String)
    open(dsl.dsl_filename, "a") do io
        write(io, instruction * "\n")
    end
end

"""
    function save_tensor_data(backend::DSLBackend,
                              node_label::Symbol,
                              tensor_label::Symbol,
                              tensor_data::Array{<:Number})

Function to save tensor data to the hdf5 file specified by a dsl backend. It
also appends a load command to the dsl script contained in the backend to load
the saved tensor data as a tensor with the given node label.
"""
function save_tensor_data(backend::DSLBackend,
                          node_label::Symbol,
                          tensor_label::Symbol,
                          tensor_data::Array{<:Number})

    # Write the tensor data to the file specified by the dsl backend
    # and append commands to the dsl script to the tensor.
    h5open(backend.tensor_data_filename, "cw") do file
        tensor_label = string(tensor_label)
        if exists(file, tensor_label)
            o_delete(file, tensor_label)
        end
        write(file, tensor_label, tensor_data)
        push!(backend, "tensor $node_label $tensor_label")
    end
end

"""
    function load_tensor_data(backend::DSLBackend,
                              tensor_label::Symbol)

Function to load tensor data from backend storage (if present)
"""
function load_tensor_data(backend::DSLBackend,
                          tensor_label::Symbol)

      if tensor_label == :result
          file = backend.output_data_filename
      else
          file = backend.tensor_data_filename
      end

      h5open(file, "r") do file
          tensor_label = string(tensor_label)
          if exists(file, tensor_label)
              return read(file, tensor_label)
          end
      end
end

"""
    function contract_tensors(backend::DSLBackend,
                              A_label::Symbol, A_ncon_indices::Array{Int, 1},
                              B_label::Symbol, B_ncon_indices::Array{Int, 1},
                              C_label::Symbol)

Function to add dsl commands to a dsl backend's script that contract two tensors
A and B to create a third tensor C.
"""
function contract_tensors(backend::DSLBackend,
                          A_label::Symbol, A_ncon_indices::Array{Int, 1},
                          B_label::Symbol, B_ncon_indices::Array{Int, 1},
                          C_label::Symbol)

    # Create and save the dsl commands for contracting A and B.
    ncon_str = "ncon " * string(C_label) * " "
    ncon_str *= string(A_label) * " " * join(A_ncon_indices, ",") * " "
    ncon_str *= string(B_label) * " " * join(B_ncon_indices, ",")
    push!(backend, ncon_str)

    # Delete the old tensors.
    push!(backend, "del " * string(A_label))
    push!(backend, "del " * string(B_label))
end

"""
    function save_output(backend::DSLBackend, node::Symbol, name::String)

Function to save the result of fully contracting a network under the given
name.
"""
function save_output(backend::DSLBackend, node::Symbol, name::String="result")
    command = "save $(node) $(backend.output_data_filename) $name"
    push!(backend, command)
end

"""
    function reshape_tensor(backend::DSLBackend,
                            tensor::Symbol,
                            groups::Array{Array{Int, 1}, 1})

Function to add dsl command that reshapes a given tensor.
"""
function reshape_tensor(backend::DSLBackend,
                        tensor::Symbol,
                        groups::Array{Array{Int, 1}, 1})
    command = "reshape $tensor " * join([join(y, ",") for y in groups], ";")
    push!(backend, command)
end


"""
    function permute_tensor(nothing, tensor::Symbol, axes)

Function to permute the axes of the given tensor
"""
function permute_tensor(backend::DSLBackend,
                        tensor::Symbol,
                        axes::Array{Int, 1})
    command = "permute $tensor " * join(axes, ",")
    push!(backend, command)
end

"""
    function decompose_tensor!(backend::DSLBackend,
                               tensor::Symbol,
                               left_indices::Array{Int, 1},
                               right_indices::Array{Int, 1};
                               threshold::AbstractFloat=1e-13,
                               max_rank::Int=0,
                               left_label::Symbol,
                               right_label::Symbol)

Function to decompose a single tensor into two tensors. Returns 0 as an
indication that the dimension of the new virtual edge cannot be determined
until runtime.
"""
function decompose_tensor!(backend::DSLBackend,
                           tensor::Symbol,
                           left_indices::Array{Int, 1},
                           right_indices::Array{Int, 1};
                           threshold::AbstractFloat=1e-13,
                           max_rank::Int=0,
                           left_label::Symbol,
                           right_label::Symbol)
    cmd_str = "decompose $tensor "
    cmd_str *= "$left_label " * join(left_indices, ",")
    cmd_str *= " $right_label " * join(right_indices, ",")
    cmd_str *= " {\"threshold\":$threshold, \"max_rank\":$max_rank}"
    push!(backend, cmd_str)
    0
end

"""
    function delete_tensor!(backend::DSLBackend, tensor_label::Symbol)

Mark tensor for deletion
"""
function delete_tensor!(backend::DSLBackend, tensor_label::Symbol)
    push!(backend, "del $tensor_label")
end

"""
    function view_tensor!(backend::DSLBackend, view_node, node, bond_idx, bond_range)

Create a view on a tensor
"""
function view_tensor!(backend::DSLBackend, view_node, node, bond_idx, bond_range)
    bond_range_str = join(bond_range, ",")
    push!(backend, "view $view_node $node $bond_idx $bond_range_str")
end
