FROM julia:1.4.2

RUN apt-get update -qq && apt-get install -y -qq gcc wget

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo PATH="/opt/conda/bin":$PATH >> /etc/bash.bashrc \
    && exec bash \
    && conda --version

ENV PYTHON /opt/conda

RUN mkdir /PicoQuant
ADD . /PicoQuant/
WORKDIR /PicoQuant

ENV JULIA_DEPOT_PATH /opt/julia 

RUN julia --project=. -e 'using Pkg; Pkg.instantiate();'

RUN julia --project=. -e 'using Pkg; Pkg.build("PicoQuant");'

# run tests to verify they are working and to precompile
RUN julia --project=. test/runtests.jl
CMD ["julia", "--project=/PicoQuant"]


