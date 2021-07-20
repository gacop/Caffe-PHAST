# Caffe-PHAST

Caffe-PHAST is a PHAST implementation of the Caffe framework (https://github.com/BVLC/caffe) developed by the University of Murcia and the University of Siena.

The PHAST Library (https://dl.acm.org/doi/abs/10.1109/TPDS.2018.2855182) is a C++ high-level library for easily programming both multi-core CPUs and NVIDIA GPUs. PHAST code can be written once and targeted to different devices via a single macro at compile time. This macro generates either CPU or NVIDIA GPU executables.

This repository contains the source code for PHAST v2 presented in the paper published in the IJHPCA journal.

## Building
First, configure your build in `Makefile.config`. You need to at least set up the `PHAST_DIR` and `PHAST_AI` variables, pointing to the correct location of the PHAST installation.

Then, build the project with `make -j$(nproc)`.

### Building for CPU
To build the project for CPUs, just run `make`, which will use the `Makefile` file.

### Building for GPU
To build the project for CPUs, `make -f Makefile-cuda`, which will use the `Makefile-cuda` file, a specific Makefile created for building for NVIDIA GPUs.


## Citation
To cite the article, use:

    @article{tobecompleted,
      Author = {},
      Journal = {},
      Title = {},
      Year = {}
    }
