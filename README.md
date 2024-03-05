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

New: please see the new `mamba`-based install script `setup.sh` for
streamlined installation.

## Free-Form Topology and AutoDiff-Enhanced Feature-Mapping Approaches

New optional experimental modules for topology optimization and automatic 
differentiation enhanced feature-mapping approaches are implemented in 
emopt/experimental, with corresponding examples in examples/experimental. 
The AutoDiff methods can result in large improvements in optimization speed for 
designs with variables that parameterize global geometric features. Please see 
our preprint below and examples for correct usage. Note: Requires PyTorch 
installation. These features are still in development.

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

Shape optimization feature-mapping methods accelerated by automatic differentiation:

S. Hooten, P. Sun, L. Gantz, M. Fiorentino, R. Beausoleil, T. Van Vaerenbergh, "Automatic Differentiation Accelerated Shape Optimization Approaches to Photonic Inverse Design on Rectilinear Simulation Grids." arXiv [cs.CE], 2311.05646 (2023). Link [here](https://arxiv.org/abs/2311.05646).
