from setuptools import setup
from distutils.command.install import install as DistutilsInstall
import subprocess

class MakeInstall(DistutilsInstall):
    def run(self):
        subprocess.call('make')
        DistutilsInstall.run(self)

setup(name='emopt',
      version='0.2.2',
      description='A suite of tools for optimizing the shape and topology of ' \
      'electromagnetic structures.',
      url='',
      author='Andrew Michaels',
      author_email='amichaels@berkeley.edu',
      license='Apache 2.0',
      packages=['emopt'],
      package_data={'':['*.csv']},
      install_requires=['numpy', 'scipy', 'mpi4py', 'petsc4py', 'slepc4py'],
      cmdclass={'install':MakeInstall},
      zip_safe=False)
