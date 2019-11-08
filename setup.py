from setuptools import setup

setup(
	name='MeasurementAutomation',
	version='0.1.0',
	author='Gleb Fedorov',
	author_email='gleb.fedorov@phystech.edu',
	url='https://github.com/vdrhtc/Measurement-automation',
	license='LICENSE',
	packages=['lib','lib2', 'lib2.tests', 'drivers'],
	package_dir = {'lib': 'lib', 'lib2': 'lib2', 'drivers': 'drivers'},
	long_description=open('README.md').read(),
	install_requires=['numpy', 'PySide2', 'scipy', 'ipython', 'matplotlib', 'tqdm', 'Cython', 'pyvisa', 'qutip', 'resonator-tools-vdrhtc', 'loggingserver'],
	zip_safe=False,
)
