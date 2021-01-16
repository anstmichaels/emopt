"""A script which downloads, compiles, and installs the dependencies needed to
install and run EMopt.

EMopt depends on 2 core open source software packages:

    3. PETSc -- solving large Ax=b problems, parallel computing
    4. SLEPc -- solving large eigenvalue problems in parallel

To run this script, simply call:

    $ python3 install_deps.py

Note: This script no longer needs to be called by the user. Instead, setup.py/pip will
take care of it for you.
"""

import os, sys, shutil, glob, requests, pathlib
from subprocess import call
from argparse import ArgumentParser

# EMopt parameters
emopt_dep_file = ".emopt_deps"

class Logger(object):
    """Setup log file."""

    def __init__(self, log_fname):
        self.terminal = sys.stdout

        # clean up old log
        if(os.path.isfile(log_fname)):
            os.remove(log_fname)

        self.log = open(log_fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def print_message(s):
    """Define custom print message which adds color."""
    print(''.join(['\033[92m', s, '\033[0m']))

def install_begin(build_dir):
    """Prepare for building dependencies.
    This primarily involves moving into the build directory.
    """
    os.chdir(build_dir)

def install_end(start_dir, build_dir):
    """Cleanup post installation, i.e. delete build dir."""
    os.chdir(start_dir)
    if(os.path.exists(build_dir)):
        shutil.rmtree(build_dir)

def write_deps_file(home_dir, include_dir, install_dir):
    """Generate the dependency file.

    The dependency file, which is stored at ./.emopt_deps, contains the paths
    of the installed dependencies. This is loaded by the setup.py script used
    to install EMopt.
    """
    dep_fname = home_dir + '/' + emopt_dep_file
    with open(dep_fname, 'w') as fdep:
        fdep.write('PETSC_DIR=' + install_dir + '\n')
        fdep.write('PETSC_ARCH=\n')
        fdep.write('SLEPC_DIR=' + install_dir + '\n')

def install_petsc(install_dir):
    """Compile and install PETSc."""
    # Clean up environment variables. If these are set the PETSc compilation will
    # fail
    if('PETSC_DIR'  in os.environ): del os.environ['PETSC_DIR']
    if('PETSC_ARCH' in os.environ): del os.environ['PETSC_ARCH']

    # get PETSc
    print_message('Downloading PETSc...')
    call(['git', 'clone', '-b', 'release', 'https://gitlab.com/petsc/petsc.git petsc'])
    call(['cd', 'petsc'])

    # compile
    print_message('Compiling PETSc...')
    call(["./configure", "--with-scalar-type=complex", "--with-mpi=1",
          "--COPTFLAGS='-O3'", "--FOPTFLAGS='-O3'", "--CXXOPTFLAGS='-O3'",  
          "--with-debugging=0", "--prefix="+install_dir, "--download-scalapack", 
          "--download-mumps", "--download-openblas"])
    call(['make', 'all'])
    call(['make', 'check'])

    print_message('Installing PETSc...')
    call(['make', 'install'])
    os.environ['PETSC_DIR'] = install_dir

    # cleanup
    print_message('Cleaning up working directory...')
    os.chdir('../')
    shutil.rmtree('petsc')

def install_slepc(install_dir):
    """Compile and install SLEPc."""
    # SLEPC_DIR environment var cant be set
    if('SLEPC_DIR' in os.environ): del os.environ['SLEPC_DIR']

    # get the SLEPc source
    print_message('Downloading SLEPc source...')
    call(['git', 'clone', '-b', 'release', 'https://gitlab.com/slepc/slepc.git slepc'])
    call(['cd', 'slepc'])

    # compile and install
    print_message('Compiling SLEPc...')
    slepc_folder = "slepc-" + SLEPC_VERSION
    os.chdir(slepc_folder)
    call(['./configure', '--prefix='+install_dir])
    call(['make', 'all'])
    call(['make', 'install'])
    call(['make', 'test'])

    # cleanup
    os.chdir('../')
    shutil.rmtree('slepc')

def install_deps(prefix=None):
    # setup logging
    sys.stdout = Logger("install.log")

    # setup install directory
    home_dir = os.path.expanduser('~')
    if(prefix == None):
        install_dir = home_dir + '/.emopt/'
    else:
        install_dir = args.prefix

    if(not os.path.exists(install_dir)):
        os.makedirs(install_dir)

    # setup installation subdirs
    include_dir = install_dir + 'include/'
    lib_dir = install_dir + 'lib/'

    if(not os.path.exists(include_dir)):
        os.makedirs(include_dir)

    if(not os.path.exists(lib_dir)):
        os.makedirs(lib_dir)

    # setup working directory
    current_dir = os.getcwd()
    build_dir = './build/'
    if(not os.path.exists(build_dir)):
        os.makedirs(build_dir)

    # save file to user's home directory which will tell emopt where to look for the
    # dependencies


    # install dependencies
    install_begin(build_dir)
    try:
        install_petsc(install_dir)
        install_slepc(install_dir)
        install_end(current_dir, build_dir)
        write_deps_file(home_dir, include_dir, install_dir)
    except Exception as e:
        print(e)
        install_end(current_dir, build_dir)

    print_message('Finished installing EMOpt dependencies!')

if __name__ == '__main__':
    # Do Argument parsing
    parser = ArgumentParser()
    parser.add_argument("--emopt-prefix=", metavar='filepath', type=str, dest='emopt_prefix', 
                        help='Set the installation directory for EMopt dependencies')

    args = parser.parse_args()

    install_deps(args.emopt_prefix)

