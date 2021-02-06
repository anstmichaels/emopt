from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, os, sys, shutil

class RunMake(install):
    def run(self):
        # Compile C++ components of EMopt
        # First check if Eigen is installed. If it isn't, we will just download it since it
        # is a header-only library
        if(not os.path.exists('/usr/include/eigen3')):
            EIGEN_VERSION = '3.3.9'
            subprocess.call(['git', 'clone', '--branch', EIGEN_VERSION, 'https://gitlab.com/libeigen/eigen.git'])
            shutil.copytree('eigen/Eigen', 'src/Eigen')
            
        subprocess.call(['make'])
        install.run(self)

def get_version_number():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(base_dir + '/emopt/__init__.py', 'r') as fin:
        for line in fin:
            if('__version__' in line):
                version = line.split('=')[-1].strip().strip().strip('\"')

    return version

setup(name='emopt',
      version=get_version_number(),
      description='A suite of tools for optimizing the shape and topology of ' \
      'electromagnetic structures.',
      url='https://github.com/anstmichaels/emopt',
      author='Andrew Michaels',
      author_email='amichaels@berkeley.edu',
      license='GPL 3.0',
      packages=find_packages(),
      package_data={'emopt':['*.so', '*.csv', 'data/*', 'solvers/*.so']},
      include_package_data=True,
      cmdclass={'install':RunMake},
      install_requires=['numpy', 'scipy', 'matplotlib', 'mpi4py', 'petsc4py', 'slepc4py'],
      zip_safe=False)
