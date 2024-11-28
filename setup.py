from setuptools import find_packages, setup

setup(
    name='tight_binding_bismuth_and_antimony_lib',
    packages=find_packages(include=['tight_binding_bismuth_and_antimony_lib']),
    version='0.1.0',
    description='Functions for a tight binding model of pure Bi, pure Sb, and Bi-Sb alloys',
    author='Johannes Bakkelund',
    install_requires=['numpy', 'pandas', 'time', 'scipy', 'joblib', 'random', 'matplotlib'],
)
