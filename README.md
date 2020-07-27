# PicoQuant

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ICHEC.github.io/PicoQuant.jl/dev)
[![Build Status](https://travis-ci.org/ICHEC/PicoQuant.jl.svg?branch=master)](https://travis-ci.org/ICHEC/PicoQuant.jl.svg?branch=master)


PicoQuant is an early prototype of a quantum circuit simulation framework being
developed as part of the [QuantEx](https://git.ichec.ie/quantex/quantex) project.
This is a [PRACE](https://prace-ri.eu/) funded project to develop quantum circuit
simulation tools capable of running on the classical exa-scale compute clusters
expected to be deployed in the coming years.

This initial prototype is written in the spirit of writing one to
throw away. The aim is to illustrate core project concepts and come face to face
with design decisions and trade-offs that will be required in the project.

The best way to get started is to:
- Follow the series of tutorial notebooks in the [notebooks folder](nb/)
- Install the prototype locally by following the instructions below
- For more details on function interfaces read the [online docs](https://ICHEC.github.io/PicoQuant.jl/dev)
- Get involved by posting an issue or submitting a pull request

## Installation and setup

The prototype comes in the form of a Julia package which is targeted to versions
of Julialang from v1 on. Julialang binaries and sources can be downloaded from
[https://julialang.org/](https://julialang.org/).

Once installed, from the Julia REPL prompt navigate to the PicoQuant folder
and activate the environment, instantiate it and then build PicoQuant.
This should install dependencies specified in the `Project.toml` file
and carry out any package specific build tasks (detailed in `deps/build.jl` file).
Currently PicoQuant uses some functionality from [qiskit](https://qiskit.org). This is
 installed in the python environment used by [PyCall](https://github.com/JuliaPy/PyCall.jl)
during the build. See below for details about using different python environments.

```
]activate .
]instantiate .
]build PicoQuant
```

## Running the unittests

Unittests can be run from the PicoQuant root folder with

```
julia --project=. test/runtests.jl
```

This will run all the unittests. It's possible to run a subset of the unittests
by passing the name of the testset. For example to run the layer3 tests contained
in the `test/layer3_tests.jl` script one would run

```
julia --project=. test/runtests.jl layer3_tests
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
On Linux systems this will use the python3 binary in the path (or python if there
is no python3 binary found). On windows and macOS systems it will create a
dedicated conda environment which will reside at `${HOME}/.julia/conda/3`.
The required python packages are installed as part of the build of PicoQuant.
To use a different python environment, the `PYTHON` environment variable must
be set to point to the python binary and PyCall needs to be (re)built. For example
to create a new conda environment at `~/.julia/conda/picoquant_env`, one would
follow the steps

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
julia --project=docs docs/make.jl
```

The documentation will be placed in the build folder and can be hosted locally
by starting a local http server with

```
cd docs/build && python3 -m http.server
```

As part of the CI this documentation is automatically built and hosted via github
pages at [https://ICHEC.github.io/PicoQuant.jl/dev](https://ICHEC.github.io/PicoQuant.jl/dev)
