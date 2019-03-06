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
EMOpt is currently released under the GPL license (see LICENSE.md for details)

## References
The methods employed by EMopt are described in:

Andrew Michaels and Eli Yablonovitch, "Leveraging continuous material averaging for inverse electromagnetic design," Opt. Express 26, 31717-31737 (2018)

An example of applying these methods to real design problems can be found in:

Andrew Michaels and Eli Yablonovitch, "Inverse design of near unity efficiency perfectly vertical grating couplers," Opt. Express 26, 4766-4779 (2018)
