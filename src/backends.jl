"Abstract type defining backends which determine the behavior of functions that
act on TensorNetworkCircuits"
abstract type AbstractBackend end

include("backends/interactive.jl")
include("backends/dsl.jl")

export save_tensor_data, load_tensor_data, save_output
export contract_tensors
export reshape_tensor, permute_tensor
export decompose_tensor!
