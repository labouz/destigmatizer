from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='reframe',
    version='0.0.1',
    description='A destigmatizing utility for langauge',
    package_dir={'': 'src'},
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/labouz/reframe',
    author='Layla Bouzoubaa, Muqi Guo, and Joseph Trybala',
    license='', #fill
    classifiers=[], #fill
    install_requires=[], #fill
    extras_require={}, #fill
    python_requires='', #fill
)


