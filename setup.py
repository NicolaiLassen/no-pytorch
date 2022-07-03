from setuptools import setup, find_packages

setup(
  name = 'no-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Neural Operator (no) - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Nicolai M. T. Lassen',
  author_email = 'mail@nmtl.dk',
  url = 'https://github.com/NicolaiLassen/no-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'fourier',
    'galerkin',
    'partial differential equations'
  ],
  install_requires=[
    'einops>=0.4.1',
    'torch>=1.10',
    'rotary_embedding_torch>=0.1.5'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 1 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)