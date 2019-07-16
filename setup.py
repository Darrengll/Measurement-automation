from setuptools import setup

setup(
	name='MeasurementAutomation',
	version='0.1.0',
	author='Gleb Fedorov',
	author_email='gleb.fedorov@phystech.edu',
	url='https://github.com/vdrhtc/Measurement-automation',
	license='LICENSE',
	scripts=['scripts/ac_stark_tts.py', 'scripts/ramsey_interative.py', 'scripts/script.py', 'scripts/script_ac_stark.py', 'scripts/script_adapt_center_freqs_for_2tone.py', 'scripts/script_powerscan.py', 'scripts/script_powerscan_ac_stark.py', 'scripts/script_powerscan_salakard.py', 'scripts/script_two_tone.py', 'scripts/script_two_tone_adaptive_frequency.py', 'scripts/script_two_tone_salakard.py'],
	packages=['lib','lib2', 'lib2.tests', 'drivers'],
	package_dir = {'lib': 'lib', 'lib2': 'lib2', 'drivers': 'drivers'},
	long_description=open('README.md').read(),
	install_requires=['visa', 'numpy', 'PySide2', 'scipy', 'ipython', 'matplotlib', 'tqdm', 'Cython', 'pyvisa', 'qutip', 'resonator-tools-vdrhtc', 'loggingserver'],
	zip_safe=False,
)
