# Layer 2 Operations

Layer 2 operations are concerned with manipulating tensor network structures
and are responsible for coordinating and passing off to layer 1 functions
to perform the computations.

## Title

```@docs
PicoQuant.random_contraction_plan
PicoQuant.inorder_contraction_plan
PicoQuant.contraction_plan_to_json
PicoQuant.contraction_plan_from_json
PicoQuant.contract_pair!
PicoQuant.contract_network!
PicoQuant.full_wavefunction_contraction!
PicoQuant.compress_tensor_chain!
PicoQuant.contract_mps_tensor_network_circuit!
PicoQuant.calculate_mps_amplitudes!
PicoQuant.sort_indices
PicoQuant.create_ncon_indices
```
