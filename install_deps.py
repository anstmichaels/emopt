"""A script which downloads, compiles, and installs the dependencies needed to
install and run EMopt.

EMopt depends on 2 core open source software packages:

    3. PETSc -- solving large Ax=b problems, parallel computing
    4. SLEPc -- solving large eigenvalue problems in parallel

To run this script, simply call:

    $ python3 install.py

By default, this will create the directory ~/.emopt and install all of the
libraries there. If you want to install these files elsewhere, you can use the
prefix flag:

    $ python3 install.py --prefix=/custom/install/path

For example, for a system-wide install, we might use:

    $ python3 install.py --prefix=/opt/local

where /opt/local is an existing directory.

If this script fails for any reason, read through the output and check the
install.log file which should be created. This should give you some indication
of what went wrong. In most cases, the issue will be related to not having the
appropriate prerequisite software packages installed.
"""

import os, sys, shutil, glob, requests, pathlib
from subprocess import call
from argparse import ArgumentParser

# EMopt parameters
emopt_dep_file = ".emopt_deps"

# Package Parameters
EIGEN_VERSION = "3.3.7"
BOOST_VERSION = "master"
PETSC_VERSION = "3.12.1"
SLEPC_VERSION = "3.12.1"


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
    base_path = os.path.dirname(os.path.realpath(__file__))
    dep_fname = base_path + '/' + emopt_dep_file
    with open(dep_fname, 'w') as fdep:
        fdep.write('EIGEN_DIR=' + include_dir + '\n')
        fdep.write('BOOST_DIR=' + include_dir + '\n')
        fdep.write('PETSC_DIR=' + install_dir + '\n')
        fdep.write('PETSC_ARCH=\n')
        fdep.write('SLEPC_DIR=' + install_dir + '\n')

def install_eigen(include_dir):
    """Download and install Eigen. It's header-only, so nice and easy."""
    print_message('Downloading Eigen headers...')
    call(['git', 'clone', 'https://github.com/eigenteam/eigen-git-mirror.git'])

    print_message('Unpacking library...')
    os.chdir('eigen-git-mirror')
    call(['git', 'checkout', EIGEN_VERSION])

    eigen_dest = include_dir + 'Eigen'
    if(os.path.exists(eigen_dest)): shutil.rmtree(eigen_dest)
    shutil.copytree('Eigen', eigen_dest)

    print_message('Cleaning up...')
    os.chdir('../')
    shutil.rmtree('eigen-git-mirror')

def install_boost(include_dir):
    """Download and install boost.geometry.

    Despite being header-only, boost is a bit tricky to install. We essentially
    acquire the whole of boost (which takes up ~1 GB) and then extract out the
    geometry library.
    """
    print_message('Retrieving boost.geometry headers. This may take a few minutes...')

    call(['git', 'clone', '--recursive', 'https://github.com/boostorg/boost.git'])
    os.chdir('boost')
    call(['git', 'checkout', BOOST_VERSION])
    call(['./bootstrap.sh'])
    call(['./b2', 'headers'])

    boost_dest = include_dir+'boost'
    boost_libs_dest = include_dir+'libs/'
    if(os.path.exists(boost_dest)): shutil.rmtree(boost_dest)
    if(os.path.exists(boost_libs_dest)): shutil.rmtree(boost_libs_dest)
    shutil.copytree('./boost', boost_dest)
    shutil.copytree('./libs', boost_libs_dest)

    print_message('Cleaning up boost directories')
    os.chdir('../')
    shutil.rmtree('boost')

def install_petsc(install_dir):
    """Compile and install PETSc."""
    # Clean up environment variables. If these are set the PETSc compilation will
    # fail
    if('PETSC_DIR'  in os.environ): del os.environ['PETSC_DIR']
    if('PETSC_ARCH' in os.environ): del os.environ['PETSC_ARCH']

    # get PETSc
    print_message('Downloading PETSc...')
    petsc_url = "http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-" + PETSC_VERSION + \
                ".tar.gz"
    petsc_fname = "petsc-" + PETSC_VERSION + ".tar.gz"
    r = requests.get(petsc_url, allow_redirects=True)
    with open(petsc_fname, 'wb') as fsave:
        fsave.write(r.content)

    # unzip package
    call(['tar', 'xvzf', petsc_fname])

    petsc_folder = "petsc-" + PETSC_VERSION
    os.chdir(petsc_folder)

    # compile
    print_message('Compiling PETSc...')
    call(["./configure", "--with-scalar-type=complex", "--with-mpi=1",
          "--COPTFLAGS='-O3'", "--FOPTFLAGS='-O3'", "--CXXOPTFLAGS='-O3'",  
          "--with-debugging=0", "--prefix="+install_dir, "--download-scalapack", 
          "--download-mumps", "--download-openblas"])
    call(['make', 'all', 'test'])

    print_message('Installing PETSc...')
    call(['make', 'install'])
    os.environ['PETSC_DIR'] = install_dir

    # cleanup
    print_message('Cleaning up working directory...')
    os.chdir('../')
    shutil.rmtree(petsc_folder)

def install_slepc(install_dir):
    """Compile and install SLEPc."""
    # SLEPC_DIR environment var cant be set
    if('SLEPC_DIR' in os.environ): del os.environ['SLEPC_DIR']

    # get the SLEPc source
    print_message('Downloading SLEPc source...')
    slepc_url = "http://slepc.upv.es/download/distrib/slepc-" + SLEPC_VERSION + \
                ".tar.gz"
    slepc_fname = "slepc-" + SLEPC_VERSION + ".tar.gz"
    r = requests.get(slepc_url, allow_redirects=True)
    with open(slepc_fname, 'wb') as fsave:
        fsave.write(r.content)

    # unzip package
    call(['tar', 'xvzf', slepc_fname])

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
    shutil.rmtree(slepc_folder)
    os.remove(slepc_fname)

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

