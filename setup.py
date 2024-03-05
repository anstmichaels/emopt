from setuptools import setup, find_packages
from setuptools.command.install import install as SetuptoolsInstall
import subprocess, os, sys

class MakeInstall(SetuptoolsInstall):
    def run(self):
        home_dir = os.path.expanduser('~')
        deps_file = home_dir + '/.emopt_deps'
        if(os.path.exists(deps_file)):
            with open(deps_file, 'r') as fdeps:
                for line in fdeps:
                    toks = line.rstrip('\r\n').split('=')
                    os.environ[toks[0]] = toks[1]
        else:
            pass # install dependencies as needed
        subprocess.call('make')
        SetuptoolsInstall.run(self)

setup(name='emopt',
      version='2023.01.16',
      description='A suite of tools for optimizing the shape and topology of ' \
      'electromagnetic structures.',
      url='https://github.com/anstmichaels/emopt',
      author='Andrew Michaels',
      author_email='amichaels@berkeley.edu',
      license='BSD-3',
      packages=find_packages(),
      package_data={'':['*.so', '*.csv', 'data/*']},
      cmdclass={'install':MakeInstall},
      install_requires=['numpy', 'scipy', 'mpi4py', 'petsc4py', 'slepc4py'],
      extras_require={"experimental": ['torch',]},
      zip_safe=False)
