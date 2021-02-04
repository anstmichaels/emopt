.. _installation_instructions:

#########################
Installation Instructions
#########################

These instructions will help you get EMopt up and running on your system.

======================
Software Prerequisites
======================

EMopt depends on a variety of software packages/libraries. Before using EMopt,
these packages must be installed:

-------------------
C/C++ Requirements:
-------------------
* `Eigen <http://eigen.tuxfamily.org/>`_: A library for handling arrays and
  matrices
* `boost.geometry <http://www.boost.org/doc/libs/develop/libs/geometry/doc/html/index.html)>`_:
  For performing computational geometry operations.
* `PETSc <https://www.mcs.anl.gov/petsc/>`_: For solving large systems of the form Ax=b.
* `SLEPc <http://slepc.upv.es/>`_: For solving large eigenvalue problems of the form Ax=nBx.

--------------------
Python Requirements:
--------------------
* `Python 2.7+ <https://www.python.org/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `mpi4py <http://mpi4py.scipy.org/docs/>`_: For handling paralellism
* `petsc4py <https://pypi.python.org/pypi/petsc4py>`_: Python interface for PETSc for solving large linear systems of
  equations
* `slepc4py <https://pypi.python.org/pypi/slepc4py>`_: Python interface for SLEPc for solving large eigenvalue
  problems.

------------------------
Optional Python Packages
------------------------

These packages are not required for the core functionality of EMopt but
provided useful functionality which is leveraged within some of EMopt's modules
(e.g. plotting, defining complex geometries, etc).

* `h5py <http://www.h5py.org/>`_: For saving and loading simulation data.
* `matplotlib <https://matplotlib.org/>`_: For plotting optimization results
* `shapely <https://github.com/Toblerity/Shapely>`_: For handling more complicated geometric operations

In addition to these specific packages, your system must be 
equipped with the ability to compile c++ source using a makefile.

==============================
Installing C/C++ Prerequisites
==============================

Before installing EMopt, a number of packages need to be installed. One Linux systems, this is
accomplished through your system's package manager. This portion of the installation has been
tested on RedHat8, CentOS 7, CentOS 8, Fedora 33, Ubuntu 18.04, and 20.04, and archlinux. If
using a linux distribution not listed, the process should be very similar.

If running on Windows, you can install EMopt in the Windows Subsystem for Linux (WSL).

----------------------
Installing on Redhat 8
----------------------

First, we need to install development tools so that we can compile the required
packages as well as openmpi, python, and tkinter::

    $ sudo yum groupinstall "Development Tools"
    $ sudo yum install openmpi openmpi-devel python3-pip python3-devel python3-tkinter python2


Once OpenMPI is installed, we need to load the appropriate module::

    $ . /etc/profile # make the module command accessible 
    $ module load mpi/openmpi-x86_64

If you use mpich instead, then you will need to choose the appropriate module name.
If you reboot, you will need to load it again unless you add this line to your
``.bashrc`` file (or equivalent).

Finally, we need to install some required python packages::

    $ pip install requests matplotlib numpy scipy mpi4py --user

NOTE: pip appears to work differently on Redhat. --user is passed on to pip installs within
setup.py

----------------------
Installing on CentOS 7
----------------------

First, we need to get access to epel-release repos which contain packages that we
will need::

    $ sudo yum install epel-release

Next, we need to install development tools so that we can compile the required
packages as well as openmpi, python, and tkinter::

    $ sudo yum groupinstall "Development Tools"
    $ sudo yum install openmpi openmpi-devel python-pip python-devel tkinter

Once OpenMPI is installed, we need to load the appropriate module::

    $ module load mpi/openmpi-x86_64

If you use mpich instead, then you will need to choose the appropriate module name.
If you reboot, you will need to load it again unless you add this line to your
``.bashrc`` file (or equivalent).

Finally, we need to install some required python packages::

    $ pip install requests matplotlib numpy scipy mpi4py --user


-----------------------
Installing on Fedora 28
-----------------------

First we need to install development tools (gcc, g++) so that we can compile the
required packages as well as openmpi, python, and tkinter::

    $ sudo yum groupinstall "Development Tools"
    $ sudo yum install gcc gcc-c++ openmpi openmpi-devel python-pip python-devel python2-tkinter

In order to use OpenMPI, we need to load the corresponding module::

    $ module load mpi/openmpi-x86_64   

If you use mpich instead, then you will need to choose the appropriate module name.
If you reboot, you will need to load it again unless you add this line to your
``.bashrc`` file (or equivalent).

Finally, we need to install a few python prerequisites (technically a few of these
should be installed by emopt's installation script, but we install them manually just
to be safe)::

    $ pip install requests matplotlib numpy scipy mpi4py --user

--------------------------
Installing on Ubuntu 18.04
--------------------------

First, we install necessary packages using ``apt-get``::

    $ sudo apt-get update
    $ sudo apt-get install build-essential gfortran openmpi-bin libopenmpi-dev git python python3 python3-dev python3-pip python3-tk

Finally, we install a few required python packages::

    $ install requests matplotlib numpy scipy mpi4py --user

---------------------------------------
Installing Remaining EMopt Dependencies
---------------------------------------

After all of the other prerequisites have been installed (per the instructions
above), clone the emopt repository and change into the emopt directory:

::

    $ git clone https://github.com/anstmichaels/emopt.git
    $ cd emopt

Once in the emopt directory, run the install script

::

    $ python install.py

This script will take a while (~10 minutes) to run and will temporarily require
around 1 GB of hard drive space (because boost). With any luck, it will complete
successfully and the emopt dependencies will be installed in your home directory
under ``~/.emopt``. If the script fails, check the terminal output and ``install.log`` file
for errors. Most likely, failure will result from not having the appropriate packages
installed.

================
Installing EMopt
================

Once the dependencies are installed, we are ready to install EMopt. If you installed
the dependencies using the install.py as described in the previous section, you can
go ahead and run the setup.py script::

    $ python setup.py install --user

Assuming this completes without error, you should be all set and ready to go!

In some scenarios, you may have installed the EMopt dependencies manually. In this
case, you need to create a file call ``~/.emopt_deps`` which contains the following
contents::

    EIGEN_DIR=/path/to/eigen/includes
    BOOST_DIR=/path/to/boost/includes
    PETSC_DIR=/path/to/petsc/installation
    SLEPC_DIR=/path/to/slepc/installation

For example, if you have made these dependencies available system wide by installing
them in the ``/opt`` folder, your ``~/.emopt_deps`` file might look like the
following::

    EIGEN_DIR=/opt/include
    BOOST_DIR=/opt/include
    PETSC_DIR=/opt/petsc/petsc-3.8.0
    SLEPC_DIR=/opt/slepc/slepc-3.8.1

After this file has been created, you should be ready to run the EMopt setup.py
script as described above.

To learn how to use EMopt, head over to the :ref:`tutorials
section<tutorials_main>` section.

======================
A Note on MPI + OpenMP
======================

By default, emopt (and its dependencies) will use OpenMP to further parallelize some
tasks. Unfortunately, on many systems the number of threads used for OpenMP will
default to the number of cores available. This is problematic when using more than
one process for MPI as emopt will try to use more threads than cores in the machine,
leading to slow performance. 

In order to avoid this, when running emopt on a single machine, it is advisable to
set the number of OpenMP threads to 1 using::

    $ export OMP_NUM_THREADS=1
    $ mpirun -n 12 python code_to_run.py

or::

    $ OMP_NUM_THREADS=1 mpirun -n 12 python code_to_run.py

If running on a network/cluster, increasing the number of threads used by OpenMP
should be fine.
