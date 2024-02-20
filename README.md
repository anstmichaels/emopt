# EMopt
A toolkit for shape (and topology) optimization of 2D and 3D electromagnetic
structures. 

EMopt offers a suite of tools for simulating and optimizing electromagnetic
structures. It includes 2D and 3D finite difference frequency domain solvers,
1D and 2D mode solvers, a flexible and *easily extensible* adjoint method
implementation, and a simple wrapper around scipy.minimize. Out of the box, it
provides just about everything needed to apply cutting-edge inverse design
techniques to your electromagnetic devices.

A key emphasis of EMopt's is shape optimization. Using boundary smoothing
techniques, EMopt allows you to compute sensitivities (i.e. gradient of a
figure of merit with respect to design variables which define an
electromagnetic device's shape) with very high accuracy. This allows you to
easily take adavantage of powerful minimization techniques in order to optimize
your electromagnetic device.

## Documentation

Details on how to install and use EMopt can be found
[on readthedocs](https://emopt.readthedocs.io/en/latest/). Check this link
periodically as the documentation is constantly being improved and examples
added.

Note: New recommended installation instructions
```bash
# Highly recommended: download mamba
# Note, you may not want to do this if you already use regular anaconda. However, no guarantees that the dependency resolution will work correctly.
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run the installer
$ bash Miniforge3-Linux-x86_64.sh

# Install the dependencies
$ mamba create --name emopt -y python=3.8 pip numpy scipy matplotlib requests h5py future eigen=3.3.7 boost=1.73.0 mpi4py openmpi petsc=*=*complex* petsc4py slepc slepc4py

# Activate the environment
$ mamba activate emopt

# If desired, for AutoDiff code, install pytorch (may instead install pytorch-cpu for lighter version without GPU compatibility)
$ mamba install -y pytorch

# Set environment variables. Open your ~/.bashrc and include the following lines
export OMP_NUM_THREADS=1
export EIGEN_DIR=~/miniforge3/envs/emopt/include/eigen3
export BOOST_DIR=~/miniforge3/envs/emopt/include/
export PETSC_DIR=~/miniforge3/envs/emopt/
export PETSC_ARCH=""
export SLEPC_DIR=~/miniforge3/envs/emopt/

# Source .bashrc
$ source ~/.bashrc

# Download EMopt
$ git clone https://github.com/anstmichaels/emopt.git
$ cd emopt

# Run setup.py
$ pip install -e .
```

## Topology and AutoDiff-Enhanced Feature Mapping Approaches

New experimental modules for topology optimization and automatic differentiation enhanced feature mapping approaches are implemented in emopt/experimental, with corresponding examples in examples/experimental. The AutoDiff methods can result in large improvements in optimization speed for designs with variables that parameterize global geometric features. Please see our preprint below and examples for correct usage. Note: Requires PyTorch installation in your local Python environment. These features are still in development.

## Authors
Andrew Michaels 

Sean Hooten (Topology and AutoDiff methods)

## License
EMOpt is currently released under the BSD-3 license (see LICENSE.md for details)

## References
The methods employed by EMopt are described in:

Andrew Michaels and Eli Yablonovitch, "Leveraging continuous material averaging for inverse electromagnetic design," Opt. Express 26, 31717-31737 (2018)

An example of applying these methods to real design problems can be found in:

Andrew Michaels and Eli Yablonovitch, "Inverse design of near unity efficiency perfectly vertical grating couplers," Opt. Express 26, 4766-4779 (2018)

Shape optimization accelerated by automatic differentiation.

S. Hooten, P. Sun, L. Gantz, M. Fiorentino, R. Beausoleil, T. Van Vaerenbergh, "Automatic Differentiation Accelerated Shape Optimization Approaches to Photonic Inverse Design on Rectilinear Simulation Grids." arXiv cs.CE, 2311.05646 (2023). Link [here](https://arxiv.org/abs/2311.05646).
