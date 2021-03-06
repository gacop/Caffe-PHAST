# Caffe-PHAST

[![DOI](https://zenodo.org/badge/382280372.svg)](https://zenodo.org/badge/latestdoi/382280372)
<img height="22px" src="https://img.shields.io/github/v/tag/gacop/Caffe-PHAST?label=Caffe-PHAST&style=flat-square">

Caffe-PHAST is a PHAST implementation of [the Caffe framework](https://github.com/BVLC/caffe) developed by the University of Murcia and the University of Siena.

[The PHAST Library](https://dl.acm.org/doi/abs/10.1109/TPDS.2018.2855182) is a C++ high-level library for easily programming both multi-core CPUs and NVIDIA GPUs. PHAST code can be written once and targeted to different devices via a single macro at compile time. This macro generates either CPU or NVIDIA GPU executables.

This repository contains the source code for PHAST v2 presented in [the paper published in the IJHPCA journal](https://journals.sagepub.com/doi/10.1177/10943420221077107).

## Building
First, configure your build in `Makefile.config`. You need to at least set up the `PHAST_DIR` and `PHAST_AI` variables, pointing to the correct location of the PHAST installation.

Then, build the project with `make -j$(nproc)`.

### Building for CPU
To build the project for CPUs, just run `make`, which will use the `Makefile` file.

### Building for GPU
To build the project for CPUs, `make -f Makefile-cuda`, which will use the `Makefile-cuda` file, a specific Makefile created for building for NVIDIA GPUs.


## Citation
To cite the article, use:

```
@article{doi:10.1177/10943420221077107,
  author = {Pablo Antonio Martínez and Biagio Peccerillo and Sandro Bartolini and José M García and Gregorio Bernabé},
  title = {{Performance portability in a real world application: PHAST applied to Caffe}},
  journal = {{The International Journal of High Performance Computing Applications}},
  year = {2022},
  month = {Mar},
  doi = {10.1177/10943420221077107},
  URL = {https://doi.org/10.1177/10943420221077107}
}
```
