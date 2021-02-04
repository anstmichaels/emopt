from setuptools import setup, find_packages
from distutils.command.build import build
import subprocess, os, sys

# Ubuntu's pip defaults to ignoring already installed packages. This will cause problems since
# PETSc needs to be installed with specific configure options and petsc4py would try to
# reinstall it...
os.environ['PIP_IGNORE_INSTALLED'] = '0'

class RunMake(build):
    def run(self):
        # Compile C++ components of EMopt
        subprocess.call(['make'])
        build.run(self)

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
      cmdclass={'build':RunMake},
      install_requires=['numpy', 'scipy', 'matplotlib', 'mpi4py', 'petsc4py', 'slepc4py'],
      zip_safe=False)
