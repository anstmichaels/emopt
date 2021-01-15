from setuptools import setup, find_packages
from setuptools.command.install import install as SetuptoolsInstall
import subprocess, os, sys

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
            if('PETSC_DIR' not in os.environ \
                or 'PETSC_ARCH' not in os.environ \
                or os.environ['PETSC_DIR'] == ''):
                try:
                    import petsc4py, slepc4py
                except ImportError:
                    print('petsc4py is not currently installed, but is required by EMopt.')
                    print('If the petsc-complex is available through your package manager ' \
                          'then it is recommended that you install the following packages:')
                    print('\t1. petsc-complex')
                    print('\t2. slepc-complex')
                    print('\t3. petsc4py-complex')
                    print('\t4. slepc4py-complex')
                    print('And then re-install EMopt.')
                    print('')
                    print('Alternatively, PETSc and SLEPc may be downloaded and compiled ' \
                          'for use with EMopt automatically now.')
                    inp = input('Proceed with installation of PETSc and SLEPc? [y/n] ')

                    ask_input = True
                    get_deps = False
                    while(ask_input):
                        if(inp.upper() == 'Y'):
                            get_deps = True
                            ask_input = False
                        elif(inp.upper() == 'N'):
                            sys.exit(0)
                        else:
                            print("Please enter 'y' to proceed with the installation " \
                                        "or 'n' to abort. ")
                            inp = input('Proceed with installation of PETSc and SLEPc? [y/n] ')

                        if(get_deps):
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
