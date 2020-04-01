from setuptools import setup
from setuptools import find_packages

setup(name='fastgae',
      description='FastGAE: Fast, Scalable and Effective Graph Autoencoders with Stochastic Subgraph Decoding',
      author='Deezer Research',
      install_requires=['networkx==2.2',
                        'numpy',
                        'scikit-learn',
                        'scipy==1.*',
                        'tensorflow==1.*'],
      package_data={'fastgae': ['README.md']},
      packages=find_packages())