from setuptools import setup
from setuptools.command.install import install as SetuptoolsInstall
import subprocess

class MakeInstall(SetuptoolsInstall):
    def run(self):
        subprocess.call('make')
        SetuptoolsInstall.run(self)

setup(name='emopt',
      version='0.2.3.1',
      description='A suite of tools for optimizing the shape and topology of ' \
      'electromagnetic structures.',
      url='https://github.com/anstmichaels/emopt',
      author='Andrew Michaels',
      author_email='amichaels@berkeley.edu',
      license='Apache 2.0',
      packages=['emopt'],
      package_data={'emopt':['*.so', 'data', 'data/*']},
      cmdclass={'install':MakeInstall},
      zip_safe=False)
