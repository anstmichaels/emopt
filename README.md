# Note: new automatic differentiation code (AutoDiff)
*Please see our preprint for a detailed description of new features*
S. Hooten, P. Sun, L. Gantz, M. Fiorentino, R. Beausoleil, T. Van Vaerenbergh, "Automatic Differentiation Accelerated Shape Optimization Approaches to Photonic Inverse Design on Rectilinear Simulation Grids." arXiv cs.CE, 2311.05646 (2023).

Link [here](https://arxiv.org/abs/2311.05646).

Several new modules and features implemented in .../emopt/experimental, with corresponding examples in .../examples/experimental.

New adjoint_method.AdjointMethod derived classes in experimental.adjoint_method.AutoDiff and experimental.adjoint_method.Topology allow one to use AutoDiff based acceleration of the adjoint method gradient calculation and topology based optimization, respectively. Must invoke the experimental solvers and grid materials in experimental.fdfd, experimental.fdtd, experimental.grid for correct usage. Please see examples in .../examples/experimental.

Note: Requires PyTorch installation in your local Python environment.
Note: These modules are still under development (particularly experimental.adjoint_method.Topology). Please post an issue or email me for further discussion or help.

For more information, please see:
S. Hooten, T. Van Vaerenbergh, M. Fiorentino, R. Beausoleil, "Accelerated Adjoint Shape Optimization Via Automatic Differentiation" in Conference on Lasers and Electro-Optics (CLEO), 2023.

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

## Authors
Andrew Michaels 

## License
EMOpt is currently released under the BSD-3 license (see LICENSE.md for details)

## References
The methods employed by EMopt are described in:

Andrew Michaels and Eli Yablonovitch, "Leveraging continuous material averaging for inverse electromagnetic design," Opt. Express 26, 31717-31737 (2018)

An example of applying these methods to real design problems can be found in:

Andrew Michaels and Eli Yablonovitch, "Inverse design of near unity efficiency perfectly vertical grating couplers," Opt. Express 26, 4766-4779 (2018)
