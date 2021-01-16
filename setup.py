from setuptools import setup, find_packages
from setuptools.command.install import install as SetuptoolsInstall
import subprocess, os, sys

if('petsc4py' not in sys.modules and 'PETSC_DIR' not in os.environ):
    os.system('export PETSC_DIR="NOT_INSTALLED"')
    os.system('export PETSC_ARCH=""')
    os.system('export SLEPC_DIR="NOT_INSTALLED"')
    PETSC_INSTALLED = False
else:
    PETSC_INSTALLED = True

class MakeInstall(SetuptoolsInstall):
    # Add a commandline argument for emopt prefix
    user_options = SetuptoolsInstall.user_options + [('emopt-prefix=', None, None)]

    def initialize_options(self):
        SetuptoolsInstall.initialize_options(self)
        self.emopt_prefix = None

    def finalize_options(self):
        SetuptoolsInstall.finalize_options(self)

    def run(self):
        # EMopt has a few import dependencies that need to be installed first. Some linux
        # distributions provide these packages in their package manager (e.g. Ubunutu,
        # and archlinux). If these packages are not provided, EMopt includes an install_deps
        # script which can take care of the installation.
        base_dir = os.path.dirname(os.path.realpath(__file__))
        deps_file = base_dir + '/.emopt_deps'
        if(os.path.exists(deps_file)):
            with open(deps_file, 'r') as fdeps:
                for line in fdeps:
                    toks = line.rstrip('\r\n').split('=')
                    os.environ[toks[0]] = toks[1]
        else:
            if(not PETSC_INSTALLED):
                print('petsc4py is not currently installed, but is required by EMopt.')
                print('It will be compiled and installed now.')
                print('If the petsc-complex is available through your package manager ' \
                      'then it is recommended that you install the following packages:')
                print('\t1. petsc-complex')
                print('\t2. slepc-complex')
                print('\t3. petsc4py-complex')
                print('\t4. slepc4py-complex')
                print('And then re-install EMopt.')
                print('')

                import install_deps
                install_deps.install_deps(self.emopt_prefix)

        subprocess.call('make')
        SetuptoolsInstall.do_egg_install(self)

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
      cmdclass={'install':MakeInstall},
      install_requires=['numpy', 'scipy', 'mpi4py', 'petsc4py', 'slepc4py'],
      zip_safe=False)
