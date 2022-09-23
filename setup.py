"""This file provides us the capability to distribute the censyn project as a package."""


from setuptools import setup, find_packages
from censyn.version.version import CENSYN_VERSION

# This grabs the requirements.txt and adds it to the install_requires for the setuptools.
with open('requirements.txt') as fp:
    install_requires = fp.read()

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='censyn',
    version=CENSYN_VERSION,
    long_description=long_description,
    url='https://gitlab.knexusresearch.com/CenSyn/censyn',
    packages=find_packages(exclude=['tests', 'conf']),
    entry_points={
        'console_scripts': [
            'censyn=censyn.__main__:command_line_start',
            'censynthesize=censyn.__main__:command_line_start'
        ]
    },
    include_package_data=True,
    install_requires=install_requires
)
