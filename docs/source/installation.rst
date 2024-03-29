.. _installation_instructions:

#########################
Installation Instructions
#########################

Here we list the new recommended installation instructions. Please see `Legacy Installation Instructions`_ below for a list of all dependencies. If you already have an existing installation of PETSc or are using an unusual Linux/hardware environment, you may want to skip to those instructions. The instructions here will assume that you are using a modern Linux distribution, and have installed common compiler tools (e.g., gcc). With a few tweaks, this will likely work for Mac and Windows based systems.

We have included a helpful interactive setup tool. Simply download the repo and run::

    $ git clone https://github.com/anstmichaels/emopt.git
    $ cd emopt
    $ . setup.sh

then, follow the prompts.

The folowing provides a description of what ``setup.sh`` does, in case you prefer a custom installation or are not using a Linux x86-64 architecture. 

First, we highly recommend you install ``mamba`` (see documentation `here <https://mamba.readthedocs.io/en/latest/index.html>`_ for more info)::

    $ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    $ . Miniforge3-Linux-x86_64.sh

.. note::

    Replace ``Linux-x86_64`` with your desired Miniforge release.

.. warning::

    If you have an existing installation of anaconda, you may not want to do this. You can try to replace ``mamba`` with ``conda`` below. However, note that the dependency resolution may not work correctly. Please also set strict channel priority, with priority given to the ``conda-forge`` channel in your ``~/.condarc``. There is a method to upgrade an existing anaconda installation to mamba `here <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#existing-conda-install-not-recommended>`_, but it is not recommended by the creators.

We may now install the dependencies::

    $ mamba create --name emopt -c conda-forge python pip numpy scipy matplotlib requests \
      h5py future eigen=3.3.7 boost=1.73.0 mpi4py openmpi petsc=*=*complex* petsc4py slepc slepc4py
    $ mamba activate emopt

.. note::

    This command installs OpenMPI for you in your local mamba environment. If you have an existing global system install that you would rather use, you will need to use the package specification ``openmpi=X.Y.*=external_*`` where ``X`` and ``Y`` are the major and minor version numbers. Please see `here <https://mpi4py.readthedocs.io/en/latest/install.html#using-conda>`_ for more info. 

.. note::

    You may also exclude ``eigen``, ``boost``, ``petsc``, and ``slepc`` from the above if you have existing installs (with given version numbers), and point to those installations with the environment variables provided below. However, if exlcuding ``petsc`` and ``slepc``, you will also need to exclude ``petsc4py`` and ``slecp4py``. See `here <https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/install.html>`_ for instructions to install ``petsc4py`` and ``slepc4py`` using ``pip``. Be sure to do this after you have created your ``emopt`` mamba environment.

EMopt now provides support for automatic differentiation enhanced feature-mapping methods and free-form topology optimziation. If you would like to use these experimental features, please also install PyTorch::

    $ mamba install -c conda-forge pytorch

.. note::

    If you do not have a GPU, you may prefer to install the CPU-only version of PyTorch, ``pytorch-cpu``, instead.

We now must set environment variables. Open your ``~/.bashrc`` and include::

    export OMP_NUM_THREADS=1

then reload with ``source ~/.bashrc`` on the command line. Now open a file in your home directory called ``~/.emopt_deps`` and include the following::

    export EIGEN_DIR=~/miniforge3/envs/emopt/include/eigen3
    export BOOST_DIR=~/miniforge3/envs/emopt/include/
    export PETSC_DIR=~/miniforge3/envs/emopt/
    export PETSC_ARCH=""
    export SLEPC_DIR=~/miniforge3/envs/emopt/

.. note::

   If you have alternate installation locations for any of the above, or used a different environment name for EMopt, please update the path accordingly. Your local environment path can be found with ``echo $CONDA_PREFIX``.

If you haven't done so you may now download EMopt, and then install::

    $ git clone https://github.com/anstmichaels/emopt.git
    $ cd emopt
    $ pip install --no-deps -vvv .

You've completed the installation!

################################
Legacy Installation Instructions
################################

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

The easiest way to install the EMopt dependencies is using the install.py script
which is included EMopt. Before running this script, however, we need to do a little
prep work. This preparation will varying depending on the linux distribution or (OS
X) you are using, however, the process is the same. Below we include instructions for
CentOS 7, Fedora 28, and Ubuntu 18.04.

The general flow is as follows:

1. Install development tools (gcc, gcc-c++, etc)
2. Intall openmpi and openmpi (or mpich) development libraries
3. Install python, pip, and tkinter
4. Install python libraries through pip: requests, matplotlib, numpy, scipy
5. (Depending on distribution) Load openmpi (or mpich) module
6. Compile and install the remaining dependencies using install.py (included with emopt)

In some situations you may wish to manually install some or all of the EMopt
dependencies. If this is the case, see the :ref:`manual installation
instructions<installation_instructions_manual>` for help on how to compile and
install the different dependencies.

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

    $ sudo apt-get install build-essential gfortran openmpi-bin libopenmpi-dev python python-dev python-pip git python-tk

Finally, we install a few required python packages::

    $ pip install requests matplotlib numpy scipy mpi4py --user

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
