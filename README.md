# PicoQuant

Initial prototype for QuantEx project written in the spirit of writing one to
throw away. Aim is to illustrate core project concepts and come face to face
with design decisions and trade-offs that will be required in the project.

## Installation and setup

The prototype comes in the form of a Julia package which is targeted to versions
of Julialang from v1 on. Binaries and source for this can be downloaded from
[https://julialang.org/](https://julialang.org/).

Once installed, from the Julia REPL prompt navigate to the PicoQuant folder
and activate the environment, instantiate it and then build PicoQuant.
This should install dependencies specified in the `Project.toml` and
`Manifest.toml` files as well as carry out any package specific build tasks
detailed in `deps/build.jl`. To use a custom python environment see the section
below on using different python environments.

```
]activate .
]instantiate .
]build PicoQuant
```

## Running the unittests

Unittests can be run from the PicoQuant root folder with

```
julia --project=. tests/runtests.jl
```

## Running standalone scripts

Standalone executable scripts are located in the `bin/` folder. As an example
we show how to run the `qasm2tng.jl` script to convert the `qft_3.qasm` file
to json format.

```
julia --project=. bin/qasm2tng.jl --qasm qft_3.qasm --output qft_3.json
```

## Starting a notebook server

Much of the prototyping and development is done in jupyter notebooks which
provides instant feedback and speeds development. To start a jupyer notebook
from the Julia REPL, enter

```
using IJulia
notebook()
```

This should open a browser window showing the home folder.

## Using different python environments
Note that PicoQuant makes use of python libraries via the PyCall.jl package.
By default this will create a dedicated conda environment which will reside
at `${HOME}/.julia/conda/3`. The required python packages are installed as
part of the build of PicoQuant. To use a different python environment, the
`PYTHON` environment variable must be set to point to the python binary and
PyCall needs to be (re)built. For example to create a new conda environment
at `~/.julia/conda/picoquant_env`, one would follow the steps

Use conda to create the environment from the command line
```
conda -p ~/.julia/conda/picoquant_env python=3.7
```

Then from the Julia REPL
```
ENV["PYTHON"] = "~/.julia/conda/picoquant_env"
]build PyCall
```

## Building the documentation

The package uses Documenter.jl to  generate html documentation from the sources.
To build the documentation, run the make.jl script from the docs folder.

```
cd docs && julia make.jl
```

The documentation will be placed in the build folder and can be hosted locally
by starting a local http server with

```
cd build && python3 -m http.server
```
