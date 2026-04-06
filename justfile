list:
    just --list --unsorted

configure CONFIG FLAGS="":
    #!/bin/bash
    set -euxo pipefail

    mkdir -p build/{{ uppercamelcase(CONFIG) }}
    pushd build/{{ uppercamelcase(CONFIG) }}
    cmake ../.. -DCMAKE_BUILD_TYPE={{ uppercamelcase(CONFIG) }} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    popd
    cp build/{{ uppercamelcase(CONFIG) }}/compile_commands.json .

build CONFIG FLAGS="": (configure CONFIG FLAGS)
    #!/bin/bash
    set -euxo pipefail

    mkdir -p build/{{ uppercamelcase(CONFIG) }}
    pushd build/{{ uppercamelcase(CONFIG) }}
    cmake --build . --config {{ uppercamelcase(CONFIG) }} --parallel
    popd

run CONFIG FLAGS="": (build CONFIG FLAGS)
    ./build/{{ uppercamelcase(CONFIG) }}/mpi_cuda_coursework

clean:
    rm -rf build slurm_build
    rm -rf .cache
    rm -f compile_commands.json

distclean: clean
    rm -rf slurm_output/*
