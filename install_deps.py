"""Install PETSc and SLEPc which are required by EMopt.

EMopt requires complex number and MUMPS support. In order to get these features, both packages
must be compiled from scratch.
"""
import subprocess, os, sys

# All arguments passed to this script will be passed on to the pip install calls below
# For example, supplying --user will install the EMopt dependencies locally and --upgrade will
# update the dependencies.
pip_args = sys.argv[1:]
pip_cmd = [sys.executable, '-m', 'pip', 'install']

# Install (or update) PETSc. If an existing PETSc installation is detected, this will be
# skipped
if('PETSC_DIR' not in os.environ or 'PETSC_ARCH' not in os.environ):
    os.environ['PETSC_CONFIGURE_OPTIONS'] = "--with-scalar-type=complex " \
                                                        "--with-mpi=1 " \
                                                        "--COPTFLAGS='-O3' " \
                                                        "--FOPTFLAGS='-O3' " \
                                                        "--CXXOPTFLAGS='-O3' " \
                                                        "--with-debugging=0 " \
                                                        "--download-scalapack " \
                                                        "--download-mumps " \
                                                        "--download-openblas"
    subprocess.call(pip_cmd + ['numpy'] + pip_args)
    subprocess.call(pip_cmd + ['petsc', '--no-binary', ':all:'] + pip_args)

# Install SLEPc. If an existing installation is detected, skip
if('SLEPC_DIR' not in os.environ):
    subprocess.call(pip_cmd + ['slepc', '--no-binary', ':all:'] + pip_args)

