# Backend Operations

Backends in PicoQuant aim to provide the same functionality but each is tailored
for a specific purpose/hardware platform/use case. Each backend should implement
the same interface. Here you will find documentation on the backend structures
and the functions they implement.

## Title

```@docs
PicoQuant.AbstractBackend
PicoQuant.DSLBackend
PicoQuant.InteractiveBackend
PicoQuant.save_tensor_data
PicoQuant.load_tensor_data
PicoQuant.save_output
PicoQuant.reshape_tensor
PicoQuant.decompose_tensor!
PicoQuant.push!
```
