"""
    QuantexQASM.jl - Initial implementation of OpenQASM generator for 
    different quantum algorithms.

# Example: QFT
We create a create register, labelled as `myReg`, and a range of qubit indices.
The `gen_qft` function will output a semi-colon delimited string of instructions
in OpenqASM format.

```julia
using QuantexQASM
q_register = "myReg"
qubit_indices = Array{Int64}(0:5)

gen_qft(q_register, qubit_indices)
```

To more easily visualise the output, we can format it for readability with
```julia
output = gen_qft(q_register, qubit_indices)
print(format_string_nl(output))
```
"""

module QuantexQASM

#Automagically export all symbols from included modules
using Reexport

include("GateOps.jl")
@reexport using .GateOps
include("QFT.jl")
@reexport using .QFT
include("Utils.jl")
@reexport using .Utils
include("VQE.jl")
@reexport using .VQE

end 
